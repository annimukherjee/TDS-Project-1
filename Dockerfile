FROM python:3.12-bookworm



# We will install a bunch of things here ------------------------------------------------
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm

# Install npx globally if not included with npm
RUN npm install -g npx

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# --------------------------------------------------------------


# Set the working directory by essentially doing `cd /app`
WORKDIR /app


RUN mkdir -p /data

# Copy the application code into the container
COPY app.py Dockerfile ./

CMD ["uv", "run", "app.py"]

