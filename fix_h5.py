import h5py

def fix_h5_model(file_path):
    try:
        with h5py.File(file_path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config = f.attrs['model_config']
                if isinstance(model_config, bytes):
                    config_str = model_config.decode('utf-8')
                    if '"batch_shape":' in config_str:
                        print(f"Fixing {file_path} (bytes)")
                        new_config = config_str.replace('"batch_shape":', '"batch_input_shape":')
                        f.attrs['model_config'] = new_config.encode('utf-8')
                else:
                    config_str = model_config
                    if '"batch_shape":' in config_str:
                        print(f"Fixing {file_path} (string)")
                        new_config = config_str.replace('"batch_shape":', '"batch_input_shape":')
                        f.attrs['model_config'] = new_config
        print(f"Verified {file_path}")
    except Exception as e:
        print(f"Error on {file_path}: {e}")

fix_h5_model("models/emotion_model.h5")
fix_h5_model("models/breed_model.h5")
