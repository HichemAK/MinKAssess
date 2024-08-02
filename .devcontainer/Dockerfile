# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM continuumio/miniconda3

# Arguments that can be set with docker build
ARG UID=1000
ARG GID=1000
ARG USERNAME=user
# Use ARG to pass the default password, but this should be overridden with --build-arg during build
ARG USER_PASSWORD=user
ARG ROOT_PASSWORD=root

# Create a group and user with specified UID and GID
RUN groupadd -g $GID usergroup && \
    useradd -m -l -u $UID -g usergroup -s /bin/bash $USERNAME

# Set the working directory in the container
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Change ownership of the workspace to the new user
RUN chown -R $USERNAME:usergroup /workspace

# Copy the environment.yml file into the container at /workspace
COPY environment.yml /workspace/environment.yml

# Set the passwords for the username and root
RUN echo "$USERNAME:$USER_PASSWORD" | chpasswd && \
    echo "root:$ROOT_PASSWORD" | chpasswd

# Switch to the new user
USER $USERNAME

# Install any needed packages specified in environment.yml
RUN conda env create -f /workspace/environment.yml

# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "KAssess", "/bin/bash", "-c"]

# # Make sure the environment is activated:
# ENV PATH /opt/conda/envs/KAssess/bin:$PATH

RUN conda init bash
