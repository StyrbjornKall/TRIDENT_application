FROM python:3.10-slim

# Create user name and home directory variables. 
# The variables are later used as $USER and $HOME. 
ENV USER=tridentapp
ENV HOME=/home/$USER
# Add user to system
RUN useradd -m -u 1000 $USER

# Set working directory (this is where the code should go)
WORKDIR $HOME/app

# Update system and install dependencies.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential

COPY pyproject.toml uv.lock $HOME/app/
COPY packages.txt $HOME/app/packages.txt

RUN apt update && apt install -y libsm6 libxext6 \
    && apt-get update && apt-get install -y libxrender-dev \
    && python -m pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . $HOME/app/

# Convert windows line endings to unix style, happens if start-script generated on windows
RUN apt-get update && apt-get install -y dos2unix \
    && dos2unix $HOME/app/start-script.sh \
    && chmod +x $HOME/app/start-script.sh \
    && chown -R $USER:$USER $HOME \
    && rm -rf /var/lib/apt/lists/*

# Serve mounts the persistent volume at $HOME/data.
# Add $HOME/data as a Mount path in Project Settings → Storage,
# then select it when creating the Custom app.
# Upload a .env file (SUPABASE_URL=..., SUPABASE_KEY=...) into the volume
# via the File Manager — it will appear at $HOME/data/.env inside the
# container. It is never baked into the image or committed to git.
ENV ENV_FILE=$HOME/data/.env

USER $USER
EXPOSE 8501

ENTRYPOINT ["./start-script.sh"]
