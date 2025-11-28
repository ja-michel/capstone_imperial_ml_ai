FROM ghcr.io/astral-sh/uv:python3.13-trixie

USER root

# Install chezmoi
ADD --chmod=755 https://get.chezmoi.io /tmp/install.sh
RUN /tmp/install.sh -b /usr/local/bin



