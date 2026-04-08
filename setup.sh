sed -i '' 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' ~/brand-gen-env/lib/python3.12/site-packages/basicsr/data/degradations.py
python3 -c "import basicsr; print('basicsr ok')"
python3 -c "import realesrgan; print('realesrgan ok')"