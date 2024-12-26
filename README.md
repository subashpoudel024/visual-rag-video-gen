To run this application, you must contain at least T4 GPU of google colab which is easily free for everyone. 
Go to google colab in your chrome browser, open a new notebook there and you have to change your runtime to T4 GPU.

Then paste this command in your notebook shell:

    !git clone https://github.com/subashpoudel024/visual-rag-video-gen.git
    !pip install gradio
    !pip install torch torchvision torchaudio  # Install PyTorch and related packages
    !pip install clip-by-openai               # Install the CLIP model from OpenAI
    !pip install faiss-cpu                   # Install FAISS (CPU version)
    !pip install numpy                       # Install NumPy
    !pip install Pillow                      # Install Pillow (PIL fork)
    !pip install scikit-learn                # Install scikit-learn for normalization
    !pip install tqdm
    !pip install git+https://github.com/openai/CLIP.git
    
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    
    print("Path to dataset files:",path)

    !python /content/visual-rag-video-gen/app.py # This command is for running the application


