import os
import urllib.request
import zipfile

zip_urls = [
    'https://d36ajqhpoeuszk.cloudfront.net/SEU-ALLEN_full_SWC_CCFv3.zip',
    'https://d36ajqhpoeuszk.cloudfront.net/SEU-ALLEN_local_SWC_CCFv3.zip',
    'https://d36ajqhpoeuszk.cloudfront.net/ION_Hipp_full_SWC_CCFv3.zip',
    'https://d36ajqhpoeuszk.cloudfront.net/ION_Hypo_full_SWC_CCFv3.zip',
    'https://d36ajqhpoeuszk.cloudfront.net/ION_PFC_full_SWC_CCFv3.zip',
    'https://d36ajqhpoeuszk.cloudfront.net/ION_PFC_full_SWC_CCFv3.zip',
    'https://d36ajqhpoeuszk.cloudfront.net/MouseLight_full_SWC_CCFv3.zip'
]

os.makedirs('SWCs', exist_ok=True)

for url in zip_urls:
    filename = url.split('/')[-1]
    zip_path = os.path.join('SWCs', filename)
    unzipped_folder = os.path.join('SWCs', filename.replace('.zip', ''))
    
    if os.path.isdir(unzipped_folder):
        print(f"Skipping {filename}, folder {unzipped_folder} already exists.")
        continue
    
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('SWCs')
        os.remove(zip_path)
        print(f"Finished {filename}.")
    except Exception as e:
        print(f"Error processing {url}: {e}")
