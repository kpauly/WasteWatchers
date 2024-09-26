import torch
import os
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, mapping
import fiona
from fiona.crs import from_epsg
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, filename='geotiff_processing.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Model paths
model_path = './RS-llava-v1.5-7b-LoRA'
model_base = 'Intel/neural-chat-7b-v3-3'

conv_mode = 'llava_v1'
disable_torch_init()
model_path = os.path.abspath(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, get_model_name_from_path(model_path))

def chat_with_RS_LLaVA(cur_prompt, image_patch):
    image_tensor = image_processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0].cuda()

    if model.config.mm_use_im_start_end:
        cur_prompt = f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}\n{cur_prompt}"
    else:
        cur_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{cur_prompt}"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], cur_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            no_repeat_ngram_size=3,
            max_new_tokens=2048,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    return outputs

def process_geotiff(tiff_path, prompt, output_shapefile):
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, NUM_THREADS='ALL_CPUS', GDAL_DISABLE_READDIR_ON_OPEN=True, GDAL_CACHEMAX=5000, GDAL_TIFF_OVR_BLOCKSIZE='1024'):
        with rasterio.open(tiff_path) as src:
            meta = src.meta.copy()
            width = meta['width']
            height = meta['height']
            transform = meta['transform']
            crs = meta['crs']
            
            schema = {
                'geometry': 'Polygon',
                'properties': {'result': 'str'},
            }

            total_tiles = (height // 224 + 1) * (width // 224 + 1)
            with fiona.open(output_shapefile, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
                with tqdm(total=total_tiles, desc=f"Processing {os.path.basename(tiff_path)}") as pbar:
                    for i in range(0, height, 224):
                        for j in range(0, width, 224):
                            window = Window(j, i, 224, 224)
                            transform_window = src.window_transform(window)
                            try:
                                # Read Band 4 to check transparency
                                band4 = src.read(4, window=window)
                                if np.all(band4 == 0):
                                    pbar.update(1)
                                    continue  # Skip this tile if all Band 4 values are 0

                                # Read the first three bands (RGB)
                                img_patch = src.read(window=window, indexes=(1, 2, 3))
                                img_patch = img_patch.transpose(1, 2, 0)
                                img_patch = Image.fromarray(img_patch.astype('uint8'), 'RGB')
                                result = chat_with_RS_LLaVA(prompt, img_patch)
                                if result.lower() == 'yes':
                                    patch_geom = box(transform_window.c, transform_window.f, transform_window.c + 224 * transform_window.a, transform_window.f + 224 * transform_window.e)
                                    shp.write({
                                        'geometry': mapping(patch_geom),
                                        'properties': {'result': result}
                                    })
                            except rasterio.errors.RasterioIOError as rio_err:
                                logging.error(f"Rasterio error at ({i}, {j}): {rio_err}")
                            except Exception as e:
                                logging.error(f"General error at ({i}, {j}): {e}")
                            pbar.update(1)

def process_folder(folder_path, prompt, num_threads):
    tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
    total_files = len(tiff_files)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_geotiff, tiff_path, prompt, f"{os.path.splitext(tiff_path)[0]}.shp"): tiff_path for tiff_path in tiff_files}
        for idx, future in enumerate(as_completed(futures), start=1):
            tiff_path = futures[future]
            try:
                future.result()
                print(f"Processed {tiff_path} successfully. ({idx} out of {total_files})")
            except Exception as e:
                print(f"Error processing {tiff_path}: {e}")

if __name__ == "__main__":
    folder_path = 'E:\\klaas\\WasteWatchers_Orthos'
    prompt = 'Is there any litter present in this image? Reply with yes or no only.'
    num_threads = 30  # Adjust based on your system's capabilities

    process_folder(folder_path, prompt, num_threads)
