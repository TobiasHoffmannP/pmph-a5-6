default: benchmark

CXX	= nvcc -O3 

RADIX_CUDA_DIR = cuda-radix
CUDA_SRC = $(RADIX_CUDA_DIR)/radix.cu
CUDA_EXECUTABLE = $(RADIX_CUDA_DIR)/radix

CUB=cub-1.8.0
RADIX_CUB_DIR = cub-code-radixsort
CUB_SRC = $(RADIX_CUB_DIR)/sorting_test.cu
CUB_EXECUTABLE = $(RADIX_CUB_DIR)/test-cub

RADIX_FUTHARK_DIR = futhark-radix
FUTHARK_SRC = $(RADIX_FUTHARK_DIR)/radix.fut
FUTHARK_EXECUTABLE = $(RADIX_FUTHARK_DIR)/radix

BENCH_HELPER = bench-helper.cu
BENCH_HELPER_EXECUTABLE = bench-helper

DATASET_SIZES = 100 1000 4000 16000 64000 1000000 4000000 64000000
DATASET_RANGE = 1000 1000 1000 1000 1000 1000 1000 1000 # Max possible value for the elements in the datasets [0:DATASET_RANGE]

benchmark: bench-futhark bench-cub bench-cuda

# bench helper
bench-helper.o: bench-helper.cu bench-helper.h
	$(CXX) -c bench-helper.cu -o bench-helper.o

# Futhark
bench-futhark:
	@echo "--Running futhark benchmark with sizes: $(DATASET_SIZES)--"
	@futhark bench --backend=cuda $(FUTHARK_SRC)

# CUB
compile-cub: $(CUB_EXECUTABLE)
$(CUB_EXECUTABLE): $(CUB_SRC) bench-helper.o
	$(CXX) -I$(CUB) -I. -o $(CUB_EXECUTABLE) $(CUB_SRC) bench-helper.o

bench-cub: compile-cub
	@echo "--Running CUB benchmark with sizes: $(DATASET_SIZES)--"
	@$(foreach i,$(shell seq 1 $(words $(DATASET_SIZES))),$(CUB_EXECUTABLE) $(word $(i),$(DATASET_SIZES)) $(word $(i),$(DATASET_RANGE));)

#CUDA
bench-cuda: compile-cuda
	@echo "--Running CUDA benchmark with sizes: $(DATASET_SIZES)--"
	@$(foreach i,$(shell seq 1 $(words $(DATASET_SIZES))),$(CUDA_EXECUTABLE) $(word $(i),$(DATASET_SIZES)) $(word $(i),$(DATASET_RANGE));)

compile-cuda: $(CUDA_EXECUTABLE)
$(CUDA_EXECUTABLE): $(CUDA_SRC) bench-helper.o
	$(CXX) -I. -o $(CUDA_EXECUTABLE) $(CUDA_SRC) bench-helper.o

gen-futhark-data:
	@mkdir -p $(RADIX_FUTHARK_DIR)/data
	@cd $(RADIX_FUTHARK_DIR) && $(foreach i,$(shell seq 1 $(words $(DATASET_SIZES))),futhark dataset --i32-bounds=0:$(if $(word $(i),$(DATASET_RANGE)),$(word $(i),$(DATASET_RANGE)),$(lastword $(DATASET_RANGE))) -b -g [$(word $(i),$(DATASET_SIZES))]i32 > data/dataset$(word $(i),$(DATASET_SIZES)).in;)

clean:
	rm -rf $(CUDA_EXECUTABLE)
	rm -rf $(CUB_EXECUTABLE)
	rm -rf bench-helper.o
	rm -rf $(RADIX_FUTHARK_DIR)/*.c $(FUTHARK_EXECUTABLE)
