FROM rust:1.72.0 as builder


RUN apt update && apt full-upgrade -y && apt install python3 python3-pip -y && apt autoremove -y
# Install libtorch
RUN pip3 install torch==2.0.0 --break-system-packages

WORKDIR /app

COPY src src
COPY benches benches
COPY build.rs build.rs
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock
ENV LIBTORCH_USE_PYTORCH=1
RUN cargo build --release --bin convert-tensor

# # ============

FROM python:3.11-slim 

COPY --from=builder /app/target/release/convert-tensor .

RUN apt update && apt full-upgrade -y && apt install libgomp1 && apt autoremove -y

RUN pip3 install torch==2.0.0 numpy==1.26.0 --break-system-packages

COPY utils utils

ENV ON_DOCKER=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/torch/lib

ENTRYPOINT [ "python3", "./utils/convert_model.py"]

