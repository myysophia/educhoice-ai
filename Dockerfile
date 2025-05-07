# 构建阶段
# 使用 Python 3.11 和 Debian Bullseye (glibc-based) 作为基础镜像
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# 为 Debian 系统安装构建依赖
# gcc, python3-dev, libffi-dev, openssl-dev 是通用构建工具
# rustc, cargo 仍可能被 chromadb 需要 (根据你之前的错误)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# 注意：--user 在这里仍然会将包安装到 /root/.local
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------------------------------------------------------------------

# 运行阶段
# 同样使用 Python 3.11 和 Debian Bullseye
FROM python:3.11-slim-bookworm

WORKDIR /app

# 对于 Debian slim 镜像, libstdc++6 通常已存在。
# 如果遇到运行时链接错误, 可以取消注释下一行来确保它被安装。
# RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制安装好的 Python 包
# Python 3.11 下 --user 安装的路径
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY vanna-mysql.py .

EXPOSE 8084

CMD ["python", "vanna-mysql.py"]