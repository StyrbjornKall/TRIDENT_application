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
    build-essential \
    software-properties-common
    
COPY requirements.txt $HOME/app/requirements.txt
COPY packages.txt $HOME/app/packages.txt
COPY start-script.sh $HOME/app/start-script.sh
COPY . $HOME/app/

RUN apt update && apt install -y libsm6 libxext6 \
    && apt-get update && apt-get install -y libxrender-dev \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && chmod +x start-script.sh \ 
    && chown -R $USER:$USER $HOME \
    && rm -rf /var/lib/apt/lists/*

# Convert windows line endings to unix style, happens if start-script generated on windows
RUN apt-get update && apt-get install -y dos2unix \
    && dos2unix $HOME/app/start-script.sh

USER $USER
EXPOSE 8501

ENTRYPOINT ["./start-script.sh"]
