# 构建阶段
FROM python:3.11-slim-bullseye AS builder

WORKDIR /build

# 安装构建依赖
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev \
    rust \
    cargo

# 复制依赖文件
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装 Python 依赖
RUN pip install --no-cache-dir --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim-bullseye

WORKDIR /app

# 只复制必要的系统库
RUN apk add --no-cache libstdc++

# 从构建阶段复制 Python 包
COPY --from=builder /root/.local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# 复制应用程序文件
COPY vanna-mysql.py .

# 暴露端口
EXPOSE 8084

# 启动应用
CMD ["python", "vanna-mysql.py"]