import os, json
import shutil

import argparse
import requests
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from copy import copy
from glob import glob
from tqdm.auto import tqdm

import diffusers, openai

import models
from models import sam

from utils.parse import parse_input_with_negative
from utils.parse import filter_boxes, draw_boxes

import generation.sdxl_refinement as sdxl

from prompt import templatev0_1 as layout_prompt_template

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def array_to_image(image):
    return Image.fromarray(image)

def image_from_array(image):
    return Image.fromarray(image)

# to get imported fn location
def get_import_path(func):
    return f"{func.__module__}.{func.__name__}"

def _read_prompt_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()#.strip()

def get_layout_prompt(prompt, layout_prompt='auto'):
    if layout_prompt == 'auto':
        template = layout_prompt_template

    elif os.path.exists(layout_prompt):
        template = _read_prompt_text(layout_prompt)

    if prompt == None:
        return template

    prompt = prompt.strip().rstrip(".")

    return template.format(prompt=prompt)

def generate_layout(prompt, client=None, layout_prompt='auto', **kwargs):

    full_prompt = get_layout_prompt(prompt, layout_prompt)

    model = kwargs.pop('prompt_model', 'gpt-4')

    if client is None or kwargs.pop('use_requests', False):
        api_key = kwargs.pop('api_key', None)
        
        if api_key is None: # check environment
            api_key = os.environ.get('OPENAI_API_KEY', None)
        
        if api_key is None: # raise ValueError
            raise ValueError('if using requests, an api_key is required')
            
        headers = {"Authorization": f"Bearer {api_key}"}

        api = f'https://api.openai.com/v1/chat/completions'

        query = {
            "model": model,
            "stop": "\n\n",
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 900,
            "temperature": 0.25,
        }

        response = requests.post(api, json=query, headers=headers)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']

    else: # use the openai client
    
        response = client.chat.completions.create(
            model=model,
            stop="\n\n",
            messages=[{"role": "user", "content": full_prompt}],

            max_tokens=900,
            temperature=0.25,
        )

        response = json.loads(response.json()) # convert

        if 'choices' in response and response['choices']:
            return response['choices'][0]['message']['content']

# replaces show_boxes in the original pipeline
def draw_layout(bboxes, bg_prompt=None, neg_prompt=None, 
                size=512, output_file=None, **kwargs):
    
    if len(bboxes) == 0:
        return None

    # Handle the size argument
    if isinstance(size, int):
        size = (size, size) 

    display = kwargs.get('show', False)

    key1, key2 = 'name', 'bounding_box'
    if not isinstance(bboxes[0], dict):
        key1, key2 = 0, 1
    
    annotations = [{'name': box[key1], 
                    'bbox': box[key2]} 
                   for _, box in enumerate(bboxes)]
    
    # Create an image with a white background
    I = np.ones((size[0] + 4, size[1] + 4, 3), dtype=np.uint8) * 255
    extent = [0, size[1], 0, size[0]]
    
    fig, ax = plt.subplots()
    ax.imshow(I)
    ax.axis('off')
    
    # Add background prompt if specified
    if bg_prompt is not None:
        prompt_text = f"{bg_prompt} (Neg: {neg_prompt})" if neg_prompt else bg_prompt
        ax.text(0, 0, prompt_text, style='italic', verticalalignment='bottom', 
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        border = plt.Rectangle((0, 0), size[1], size[0], linewidth=2, 
                               edgecolor='black', facecolor='none')
        
        ax.add_patch(border) # add the rectangular image border

    draw_boxes(annotations)

    if output_file is not None:
        plt.savefig(output_file)
    
    if display: 
        plt.show()

    plt.close(fig)

    return fig, ax # to modify further

def _parse_output_files(output_dir, n_samples, prefix, ext, 
                        add_samples=False, pad=3, **kwargs):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    existing_files = glob(os.path.join(output_dir, f"{prefix}_*.{ext}"))

    # Determine the starting index for new files
    last_file_index = 0
    if existing_files:
        last_file_index = max(int(file.split('_')[-1].split('.')[0]) for file in existing_files)

    file_indices = []
    
    if add_samples:
        file_indices = range(last_index + 1, last_index + 1 + n_samples)
    else:
        if not len(existing_files) >= n_samples:
            file_indices = range(last_file_index+1, n_samples + 1)

    file_stem = os.path.join(output_dir, prefix)

    output_files = [f'{file_stem}_{str(index).zfill(pad)}.{ext}'
                    for index in file_indices]

    return None if not output_files else output_files

def load_lmd_plus(**kwargs):
    models.sd_key = "gligen/diffusers-generation-text-box"
    models.sd_version = "sdv1.4"

    models.model_dict = models.load_sd(key=models.sd_key,
                                       use_fp16=False)

    
    sam_model_dict = sam.load_sam()
    models.model_dict.update(sam_model_dict)

    from generation import lmd_plus

    return lmd_plus # the default model

def generate(prompt, n_samples=1, layout=None, output_dir=None, 
             client=None, all_quiet=False, **kwargs):

    default_kwargs = {'run_model': 'lmd_plus',
                      'prompt_model': 'gpt-4',
                      'layout_prompt': 'auto',
                      'seed_offset': 0,
                      'use_sdxl': False,
                      'report_steps': True,
                      'progress_bars': True}

    args = argparse.Namespace(**default_kwargs)

    if layout is None: # generate a layout
        layout_prompt = args.layout_prompt
        layout = generate_layout(prompt, client, layout_prompt)

    if output_dir is not None:
        
        if kwargs.pop('fresh_start', False):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exists_ok=False)

        parse_args = (output_dir, n_samples, 'sample', 'png')
        output_files = _parse_output_files(*parse_args, **kwargs)

        if not output_files or len(output_files) == 0:
            print('All samples generated.'); return

    parsed_layout = parse_input_with_negative(layout, no_input=True)
    gen_boxes, bg_prompt, neg_prompt = parsed_layout
    
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=False)
    
    specs = {
        "prompt": prompt,
        "gen_boxes": gen_boxes,
        "bg_prompt": bg_prompt,
        "extra_neg_prompt": neg_prompt,
    }

    generation = kwargs.get('generator', None)
    if generation is None:
        generation = load_lmd_plus()
    
    version = generation.version
    run_model = generation.run

    if args.use_sdxl:
        # Offloading model saves GPU
        sdxl.init(offload_model=True)

    # constants for seeds:
    LARGE_CONSTANT1 = 123456789
    LARGE_CONSTANT2 = 56789
    LARGE_CONSTANT3 = 6789
    LARGE_CONSTANT4 = 7890

    run_kwargs = {'frozen_step_ratio': 0.5}

    for key, value in default_kwargs.items():
        if key == 'report_steps' and value:
            run_kwargs['report_steps'] = True

        if key =='progress_bars':
            if not value:
                run_kwargs['show_progress'] = False

    all_outputs = [] # +1 image per sample

    if output_dir is not None: # update
        n_samples = len(output_files)

    if not all_quiet: # report samples, specs
        print(f'Now generating {n_samples} samples with layout:\n', specs)

    for sample_index in range(n_samples):
        # This ensures different repeats have different seeds.
        index_offset = sample_index * LARGE_CONSTANT3 + args.seed_offset
    
        output = run_model(
            spec=specs,
            bg_seed=index_offset,
            fg_seed_start=index_offset + LARGE_CONSTANT1,
            overall_prompt_override=prompt,
            **run_kwargs,
        )
    
        output = output.image
    
        if args.use_sdxl:
            refine_seed = index_offset + LARGE_CONSTANT4
            
            output = sdxl.refine(image=output, spec=specs, 
                                 refine_seed=refine_seed, 
                                 refinement_step_ratio=0.3)

        if output_dir is not None:
            output_file = output_files[sample_index]
            image_from_array(output).save(output_file)

            save_json(specs, output_file.replace('png', 'json'))

        all_outputs += [output]

        return output if len(all_outputs) == 1 else all_outputs

def load_generator(args=None, run_model='lmd_plus', use_sdv2=False, **kwargs):
    if args is None:
        args = argparse.Namespace(**{'run_model': run_model, 'use_sdv2': use_sdv2})
    
    if args.use_sdv2:
        if args.run_model not in ["gligen", "lmd_plus"]:
            raise ValueError("gligen only supports SDv1.4")
            
        # We abbreviate v2.1 as v2
        models.sd_key = "stabilityai/stable-diffusion-2-1-base"
        models.sd_version = "sdv2"
        
    else:
        if args.run_model in ["gligen", "lmd_plus"]:
            models.sd_key = "gligen/diffusers-generation-text-box"
            models.sd_version = "sdv1.4"
        else:
            models.sd_key = "runwayml/stable-diffusion-v1-5"
            models.sd_version = "sdv1.5"
    
    print(f"Using SD: {models.sd_key}")
    if args.run_model not in ["multidiffusion"]:
        models.model_dict = models.load_sd(
            key=models.sd_key,
            use_fp16=False,
            # modified default:
            scheduler_cls=None,
        )

    if args.run_model in ["lmd", "lmd_plus"]:
        sam_model_dict = sam.load_sam()
        models.model_dict.update(sam_model_dict)
    
    if args.run_model == "lmd_plus":
        import generation.lmd_plus as generation
        
    elif args.run_model == "lmd":
        import generation.lmd as generation
        
    elif args.run_model == "sd":
        if not args.ignore_negative_prompt:
            print("**Running SD without `ignore_negative_prompt`.",
                  "(This means it is no longer a valid baseline.)")
        import generation.stable_diffusion_generate as generation
        
    elif args.run_model == "multidiffusion":
        import generation.multidiffusion as generation
        
    elif args.run_model == "backward_guidance":
        import generation.backward_guidance as generation
        
    elif args.run_model == "boxdiff":
        import generation.boxdiff as generation
        
    elif args.run_model == "gligen":
        import generation.gligen as generation
        
    else: # run_model is not available
        raise ValueError(f"Unknown model type: {args.run_model}")

    return generation # backend for generating images

def generate_plus(prompt, n_samples=1, layout=None, output_dir=None,
                  client=None, all_quiet=False, **kwargs):

    core_kwargs = {'run_model': 'lmd_plus',
                   'prompt_model': 'gpt-4',
                   'layout_prompt': 'auto',
                   'seed_offset': 0,
                   'use_sdv2': False,
                   'use_sdxl': False,
                   'sdxl_step_ratio': 0.3,
                   'run_kwargs': {'frozen_step_ratio': 0.5},
                   'synthetic_prompt': True}

    save_kwargs = {'fresh_start': False,
                   'add_samples': False,
                   'save_layouts': True}

    report_kwargs = {'verbose': False,
                     'reports': True,
                     'progress_bars': True}

    parsed_kwargs = kwargs.copy()

    for kwarg_set in [core_kwargs, save_kwargs, report_kwargs]:
        for key1 in kwarg_set.keys():
            for key2, value in kwargs.items():
                if key1 == key2:
                    kwarg_set[key1] = parsed_kwargs.pop(key2)

    if kwargs.pop('check_kwargs', False):
        return core_kwargs, other_kwargs, report_kwargs

    args = argparse.Namespace(**core_kwargs)

    if layout is None: # generate a layout
        layout_prompt = args.layout_prompt
        layout = generate_layout(prompt, client, layout_prompt)

    if output_dir is not None:
        save_layout = save_kwargs.pop('save_layouts')
        
        if save_kwargs.pop('fresh_start', False):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exists_ok=False)

        save_args = (output_dir, n_samples, 'sample', 'png')
        output_files = _parse_output_files(*save_args, **save_kwargs)

        if not output_files or len(output_files) == 0:
            print('All samples generated.'); return

    parsed_layout = parse_input_with_negative(layout, no_input=True)
    gen_boxes, bg_prompt, neg_prompt = parsed_layout
    
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=False)
    
    specs = {
        "prompt": prompt,
        "gen_boxes": gen_boxes,
        "bg_prompt": bg_prompt,
        "extra_neg_prompt": neg_prompt,
    }
        
    generation = kwargs.get('generator', None)
    if generation is None:
        generation = load_generation(args)
    
    version = generation.version
    run = generation.run
    
    if args.use_sdv2:
        version = f"{version}_sdv2"

    if args.use_sdxl:
        # Offloading model saves GPU
        sdxl.init(offload_model=True)

    # constants for seeds:
    LARGE_CONSTANT1 = 123456789
    LARGE_CONSTANT2 = 56789
    LARGE_CONSTANT3 = 6789
    LARGE_CONSTANT4 = 7890

    run_func_path = get_import_path(run)
    run_kwargs = args.run_kwargs.copy()

    for key, value in report_kwargs.items():
        if key == 'reports' and value:
            run_kwargs['report_steps'] = True

        if key =='progress_bars':
            if not value:
                run_kwargs['show_progress'] = False

    all_outputs = [] # +1 image per sample

    if output_dir is not None: # update
        n_samples = len(output_files)

    if not all_quiet: # report samples, specs
        print(f'Now generating {n_samples} samples with layout:\n', specs)
        print(f'Running {run_func_path} as the generator...')

    for sample_index in range(n_samples):
        # This ensures different repeats have different seeds.
        index_offset = sample_index * LARGE_CONSTANT3 + args.seed_offset
    
        if args.run_model in ['lmd', 'lmd_plus']:
            # Our models load `extra_neg_prompt` from the spec
            if not args.synthetic_prompt:
                print('Running without synthetic prompt.')
                output = run(
                    spec=specs,
                    bg_seed=index_offset,
                    fg_seed_start=index_offset + LARGE_CONSTANT1,
                    overall_prompt_override=prompt,
                    **run_kwargs,
                )
                
            else:
                # Uses synthetic prompt (handles negation and additional languages better)
                output = run(
                    spec=specs,
                    bg_seed=index_offset,
                    fg_seed_start=index_offset + LARGE_CONSTANT1,
                    **run_kwargs,
                )
                
        elif args.run_model == "sd":
            output = run(
                prompt=prompt,
                seed=index_offset,
                extra_neg_prompt=neg_prompt,
                **run_kwargs,
            )
            
        elif args.run_model == "multidiffusion":
            key = 'multidiffusion_bootstrapping'
            bootstrapping = run_kwargs.get(key, 20)
            
            output = run(
                gen_boxes=gen_boxes,
                bg_prompt=bg_prompt,
                original_index_base=index_offset,
                bootstrapping=bootstrapping,
                extra_neg_prompt=neg_prompt,
                **run_kwargs,
            )
            
        elif args.run_model == "backward_guidance":
            output = run(
                spec=specs,
                bg_seed=index_offset,
                **run_kwargs,
            )
            
        elif args.run_model == "boxdiff":
            output = run(
                spec=specs,
                bg_seed=index_offset,
                **run_kwargs,
            )
            
        elif args.run_model == "gligen":
            output = run(
                spec=specs,
                bg_seed=index_offset,
                **run_kwargs,
            )
    
        output = output.image
    
        if args.use_sdxl:
            refine_seed = index_offset + LARGE_CONSTANT4
            refinement_step_ratio = args.sdxl_step_ratio
            
            output = sdxl.refine(image=output, spec=specs, refine_seed=refine_seed, 
                                 refinement_step_ratio=refinement_step_ratio)

        if output_dir is not None:
            output_file = output_files[sample_index]
            image_from_array(output).save(output_file)

            save_json(specs, output_file.replace('png', 'json'))

        all_outputs += [output]

        return output if len(all_outputs) == 1 else all_outputs