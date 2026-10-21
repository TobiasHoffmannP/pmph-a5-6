default: benchmark

CXX	= nvcc -O3 

RADIX_CUDA_DIR = cuda-radix
CUDA_SRC = $(RADIX_CUDA_DIR)/radix.cu
CUDA_EXECUTABLE = $(RADIX_CUDA_DIR)/radix
BLOCK_SIZE = 2
NUM_OF_BITS = 2
ELM_PER_THREAD = 4

CUB=cub-1.8.0
RADIX_CUB_DIR = cub-code-radixsort
CUB_SRC = $(RADIX_CUB_DIR)/sorting_test.cu
CUB_EXECUTABLE = $(RADIX_CUB_DIR)/test-cub

RADIX_FUTHARK_DIR = futhark-radix
FUTHARK_SRC = $(RADIX_FUTHARK_DIR)/radix.fut
FUTHARK_EXECUTABLE = $(RADIX_FUTHARK_DIR)/radix

DATASET_SIZES = 10 100 10000 100000000

benchmark: bench-futhark bench-cub bench-cuda


# Futhark
bench-futhark:
	@echo "--Running futhark benchmark with sizes: $(DATASET_SIZES)--"
	@futhark bench --backend=cuda $(FUTHARK_SRC)

# CUB
compile-cub: $(CUB_EXECUTABLE)
$(CUB_EXECUTABLE): $(CUB_SRC)
	$(CXX) -o $(CUB_EXECUTABLE) $(CUB_SRC)

bench-cub: compile-cub
	@echo "--Running CUB benchmark with sizes: $(DATASET_SIZES)--"
	@$(foreach size, $(DATASET_SIZES), $(CUB_EXECUTABLE) $(size);)

#CUDA
bench-cuda: compile-cuda
	@echo "--Running CUDA benchmark with sizes: $(DATASET_SIZES)--"
	@$(foreach size,$(DATASET_SIZES),$(CUDA_EXECUTABLE) $(size) $(BLOCK_SIZE) $(NUM_OF_BITS) $(ELM_PER_THREAD);)

compile-cuda: $(CUDA_EXECUTABLE)
$(CUDA_EXECUTABLE): $(CUDA_SRC)
	$(CXX) -o $(CUDA_EXECUTABLE) $(CUDA_SRC)


gen-futhark-data:
	@cd $(RADIX_FUTHARK_DIR) && 
		$(foreach size, $(DATASET_SIZES), futhark dataset --i32-bounds=0:$(shell echo $$(( $(size) / 10 ))) -b -g [$(size)]i32 > data/dataset$(size).in;)

clean:
	rm -rf $(CUDA_EXECUTABLE)
	rm -rf $(CUB_EXECUTABLE)
	rm -rf $(RADIX_FUTHARK_DIR)/*.c $(FUTHARK_EXECUTABLE)
