ARG base_tag=latest
ARG organization=intel

FROM ghcr.io/$organization/llvm/ubuntu2004_base:$base_tag

ARG compute_runtime_version=latest
ARG igc_version=latest
ARG tbb_version=latest
ARG fpgaemu_version=latest
ARG cpu_version=latest

RUN apt update && apt install -yqq curl wget

# Install IGC
RUN curl -s https://api.github.com/repos/intel/intel-graphics-compiler/releases/$igc_version \
  | grep "browser_download_url.*deb" \
  | cut -d : -f 2,3 \
  | tr -d \" \
  | wget -qi - && \
  dpkg -i *.deb && rm *.deb

# Install NEO
RUN curl -s https://api.github.com/repos/intel/compute-runtime/releases/$compute_runtime_version \
  | grep "browser_download_url.*deb" \
  | cut -d : -f 2,3 \
  | tr -d \" \
  | wget -qi - && \
  dpkg -i *.deb && rm *.deb

RUN mkdir /runtimes

# Install TBB
RUN cd /runtimes && \
  curl -s https://api.github.com/repos/oneapi-src/onetbb/releases/$tbb_version \
  | grep -E "browser_download_url.*-lin.tgz" \
  | cut -d : -f 2,3 \
  | tr -d \" \
  | wget -qi - && \
  tar -xf *.tgz && rm *.tgz && mv oneapi-tbb-* oneapi-tbb

# Install Intel FPGA Emulator
RUN cd /runtimes && \
  curl -s https://api.github.com/repos/intel/llvm/releases/$fpgaemu_version \
  | grep -E "browser_download_url.*fpgaemu.*tar.gz" \
  | cut -d : -f 2,3 \
  | tr -d \" \
  | wget -qi - && \
  mkdir fpgaemu && tar -xf *.tar.gz -C fpgaemu && rm *.tar.gz && \
  echo  /runtimes/fpgaemu/x64/libintelocl_emu.so > /etc/OpenCL/vendors/intel_fpgaemu.icd

# Install Intel OpenCL CPU Runtime
RUN cd /runtimes && \
  curl -s https://api.github.com/repos/intel/llvm/releases/$cpu_version \
  | grep -E "browser_download_url.*oclcpuexp.*tar.gz" \
  | cut -d : -f 2,3 \
  | tr -d \" \
  | wget -qi - && \
  mkdir oclcpu && tar -xf *.tar.gz -C oclcpu && rm *.tar.gz && \
  echo  /runtimes/oclcpu/x64/libintelocl.so > /etc/OpenCL/vendors/intel_oclcpu.icd

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]
