#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))
#define BIT_SET(var,pos) ((var) |= (1ULL<<(pos)))
#define CALC_OFFSET (var,pos) (__popc((var) & (0xffffffff>>(32-pos))))

__device__ __constant__ int d_min_sup;
__device__ __constant__ int d_unique_item_count;

//function for tracking time of various operations 
void trackTime(int operation_type) {
	static std::chrono::time_point<std::chrono::steady_clock> start;
	static std::chrono::time_point<std::chrono::steady_clock> stop;
	static std::chrono::microseconds duration[7];
	static std::ofstream timefile("time.txt");
	stop = std::chrono::high_resolution_clock::now();
	switch (operation_type) {
		//load data from disc to RAM
		case 0:
			duration[0] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//transfer data from RAM to VRAM and memory allocation
		case 1:
			duration[1] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//scan database
		case 2:
			duration[2] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//prefix sum
		case 3:
			duration[3] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//stream compaction
		case 4:
			duration[4] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//transfer data from VRAM to RAM
		case 5:
			duration[5] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//save data on disc
		case 6:
			duration[6] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//other GPU operations
		case 7:
			duration[7] += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			break;
		//write time measurements to file
		case 8:
			if (timefile.is_open()) {
				for (int i = 0; i < 8/*sizeof(duration) / sizeof(duration[0])*/; i++) {
					timefile << duration[i].count() << "\n";
				}
				timefile.close();
			}
			else {
				std::cout << "Unable to open time file";
				exit(1);
			}
			break;
		//time tracking initialization - do nothing 
		case 9:
			break;

		default:
			std::cout << "Invalid value passed to trackTime function";
			exit(1);
			break;
	}
	start = std::chrono::high_resolution_clock::now();
}

inline void checkError(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
}

__global__ void firstScan(char* database, int* row_indexes_start, int* seq_val, int* new_row_idx_start, int* database_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //database row
	int local_database_size = database_size[0];

	if (i < local_database_size) {
		unsigned int seq = 0xffffffff;
		unsigned int already_found_seq = 0x0;
		unsigned int seq_candidate;

		int j = row_indexes_start[i];
		char local_char = database[j];
		while (local_char != '.') {
			if (local_char != 44) {
				seq_candidate = CHECK_BIT(seq, local_char - 97);
				seq_candidate = already_found_seq | seq_candidate;
				if (seq_candidate != already_found_seq) {
					atomicAdd(&seq_val[local_char - 97], 1);
					new_row_idx_start[(local_char - 97) * local_database_size + i] = j + 1;
					already_found_seq = seq_candidate;
				}
			}
			j++;
			local_char = database[j];
		}
	}
}

__global__ void prepareVector(int* new_row_idx_start, int* target_index, int* seq_val, int* database_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //database row
	int local_database_size = database_size[0];
	if (i < local_database_size) {
		for (int j = 0; j < d_unique_item_count; j++) { 
			target_index[j * local_database_size + i] = new_row_idx_start[j * local_database_size + i] > 0 && seq_val[j] >= d_min_sup ? 1 : 0;
		}
	}
}

__global__ void prepareVector2(int* new_row_idx_start, int* target_index, int* seq_val, int* database_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //database row
	if (i < database_size[0] && new_row_idx_start[i] > 0) {
		target_index[i] = seq_val[target_index[i]] >= d_min_sup ? 1 : 0;
	}
}

__global__ void streamCompaction(int* new_row_idx_start, int* target_index, int* row_indexes_start, int* database_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < database_size[0]) {
		if (i == 0) {
			if (target_index[i] > 0) {
				row_indexes_start[target_index[i] - 1] = new_row_idx_start[i];
			}
		}
		else if (target_index[i] > target_index[i - 1]) {
			row_indexes_start[target_index[i] - 1] = new_row_idx_start[i];
		}
	}
}

__global__ void prepareSeq(int* seq_val, int* target_index, int* seq_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < seq_size[0]) {
		target_index[i] = seq_val[i] >= d_min_sup ? 1 : 0;
	}
}

__global__ void trimSeq(char* seq, int* seq_val, char* seq_trimmed, int* seq_val_trimmed, int* target_index, int* sup_seq_idx, int* old_sup_seq_idx, int* single_seq_size, int* seq_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < seq_size[0]) {
		if (i == 0) {
			if (target_index[i] > 0) {
				for (int j = 0; j < single_seq_size[0]; j++) {
					seq_trimmed[(target_index[i] - 1) * single_seq_size[0] + j] = seq[i * single_seq_size[0] + j];
				}
				seq_val_trimmed[target_index[i] - 1] = seq_val[i];
				old_sup_seq_idx[target_index[i] - 1] = sup_seq_idx[i];
			}
		}
		else if (target_index[i] > target_index[i - 1]) {
			for (int j = 0; j < single_seq_size[0]; j++) {
				seq_trimmed[(target_index[i] - 1) * single_seq_size[0] + j] = seq[i * single_seq_size[0] + j];
			}
			seq_val_trimmed[target_index[i] - 1] = seq_val[i];
			old_sup_seq_idx[target_index[i] - 1] = sup_seq_idx[i];
		}
	}
}

__global__ void firstGenerateNewSeq(char* seq, char* seq_trimmed, int* seq_val_trimmed, int* sup_seq_idx, int* single_seq_size, unsigned int* append,
	unsigned int* assemble, unsigned int* last_element, int* sup_database_size, int* seq_val_idx, int* new_row_idx_size, int* seq_val_trimmed_size) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int seq_idx = 0;

	//Determine where given thread should write their sequences:
	for (int j = 0; j < i; j++) {
		seq_idx += 2 * seq_val_trimmed_size[0] - 1 - j;
	}

	seq_val_idx[i] = seq_idx;

	//create new sequences:
	//create appends
	for (int j = 0; j < seq_val_trimmed_size[0]; j++) {
		seq[seq_idx * single_seq_size[0]] = seq_trimmed[i];
		seq[seq_idx * single_seq_size[0] + 1] = ',';
		seq[seq_idx * single_seq_size[0] + 2] = seq_trimmed[j];
		BIT_SET(append[i], seq_trimmed[j] - 97);
		sup_seq_idx[seq_idx] = i;
		seq_idx++;
	}

	//create assemblages
	for (int j = i + 1; j < seq_val_trimmed_size[0]; j++) {
		seq[seq_idx * single_seq_size[0]] = seq_trimmed[i];
		seq[seq_idx * single_seq_size[0] + 1] = '_';
		seq[seq_idx * single_seq_size[0] + 2] = seq_trimmed[j];
		BIT_SET(assemble[i], seq_trimmed[j] - 97);
		sup_seq_idx[seq_idx] = i;
		seq_idx++;
	}

	BIT_SET(last_element[i], seq_trimmed[i] - 97);
	sup_database_size[i] = seq_val_trimmed[i];
	new_row_idx_size[i + 1] = seq_val_trimmed[i] * (2 * seq_val_trimmed_size[0] - i - 1);

}

__global__ void scanDatabase(int* database_size, int* sub_database_size, unsigned int* append, unsigned int* assemble, int* row_indexes_start,
	char* database, int* seq_val, int* new_row_idx_start, int* seq_val_idx, unsigned int* last_element, int* new_database_start_idx, int* target_index, int* seq_val_trimmed_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < database_size[0]) {
		int local_sub_database_start;
		int local_sub_database_size;
		bool comma_found = false;
		int min_boundary = 0;
		int max_boundary = seq_val_trimmed_size[0];
		int middle_point;
		
		//determine which sub database given thread is searching
		//special case - first sub database:
		if (i < sub_database_size[0]) {
			middle_point = 0;
			local_sub_database_size = sub_database_size[middle_point];
			local_sub_database_start = 0;
		}
		//standard case, locate database using binary search:
		else {
			while (true) {
				middle_point = (min_boundary + max_boundary) / 2;
				if (i >= sub_database_size[middle_point]) {
					if (i < sub_database_size[middle_point + 1]) {
						local_sub_database_size = sub_database_size[middle_point + 1] - sub_database_size[middle_point];
						local_sub_database_start = sub_database_size[middle_point];
						middle_point++;
						break;
					}
					else {
						min_boundary = middle_point;
					}
				}
				else {
					max_boundary = middle_point;
				}
			}
		}

		//find appends
		unsigned int local_seq = append[middle_point];
		unsigned int already_found_seq = 0x0;
		unsigned int seq_candidate;
		int l = row_indexes_start[i];
		char local_char = database[l]; 
		int idx_offset;
		int append_idx_offset = __popc(local_seq);

		while (local_char != '.') {
			if (local_char == 44) {
				comma_found = true;
			}
			else {
				seq_candidate = CHECK_BIT(local_seq, local_char - 97);
				seq_candidate = already_found_seq | seq_candidate;
				if (seq_candidate != already_found_seq && comma_found == true) {
					idx_offset = __popc((local_seq) & (0xffffffff >> (32 - local_char + 97)));
					atomicAdd(&seq_val[seq_val_idx[middle_point] + idx_offset], 1);
					new_row_idx_start[idx_offset * local_sub_database_size + i - local_sub_database_start + new_database_start_idx[middle_point]] = l + 1;
					target_index[idx_offset * local_sub_database_size + i - local_sub_database_start + new_database_start_idx[middle_point]] = seq_val_idx[middle_point] + idx_offset;
					already_found_seq = seq_candidate;
				}
			}
			l++;
			local_char = database[l];
		}

		//find assemblages
		l = row_indexes_start[i];
		local_char = database[l];
		already_found_seq = 0x0;
		local_seq = assemble[middle_point];
		unsigned int local_prefix = last_element[middle_point];
		unsigned int found_prefix = local_prefix;

		while (local_char != '.') {
			if (local_char == 44) { //if comma is encountered prefix needs to be found again
				found_prefix = 0x0;
			}
			else {
				seq_candidate = CHECK_BIT(local_seq, local_char - 97);
				seq_candidate = already_found_seq | seq_candidate; 
				found_prefix = found_prefix | CHECK_BIT(local_prefix, local_char - 97);
				if (seq_candidate != already_found_seq && found_prefix == local_prefix) {
					idx_offset = __popc((local_seq) & (0xffffffff >> (32 - local_char + 97))) + append_idx_offset;
					atomicAdd(&seq_val[seq_val_idx[middle_point] + idx_offset], 1);
					new_row_idx_start[idx_offset * local_sub_database_size + i - local_sub_database_start + new_database_start_idx[middle_point]] = l + 1;
					target_index[idx_offset * local_sub_database_size + i - local_sub_database_start + new_database_start_idx[middle_point]] = seq_val_idx[middle_point] + idx_offset;
					already_found_seq = seq_candidate;
				}
			}
			l++;
			local_char = database[l];
		}
	}
}

__global__ void correctOldSupSeqIdx(int* old_sup_seq_idx, int* adj_diff) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//Correct old superior sequence index values so that difference between adjecent values is <= 1
	if (i == 0) {
		adj_diff[i] = old_sup_seq_idx[i];
	}
	else {
		adj_diff[i] = old_sup_seq_idx[i] - old_sup_seq_idx[i - 1] > 1 ? old_sup_seq_idx[i] - old_sup_seq_idx[i - 1] - 1 : 0;
	}
}

__global__ void calculateNewSeqPos(char* seq_trimmed, int* seq_start, int* assemble_start, int* old_sup_seq_idx, int* single_seq_size, int* seq_val_trimmed_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//establish how many sequences have the same sup_seq_idx and where assemblages begin
	if (i < seq_val_trimmed_size[0]) {
		//special code for first sequence
		if (i == 0) {
			seq_start[i] = i;
			//check if this sequence is an assemble
			if (seq_trimmed[(i + 1) * (single_seq_size[0] - 2) - 2] == '_') {
				assemble_start[old_sup_seq_idx[i]] = i;
			}
			//check if there is only one sequence in database
			if (i == seq_val_trimmed_size[0] - 1) {
				seq_start[old_sup_seq_idx[i] + 1] = i + 1;
				if (seq_trimmed[(i + 1) * (single_seq_size[0] - 2) - 2] == ',') {
					assemble_start[old_sup_seq_idx[i]] = i + 1;
				}
			}
		}
		else {
			//special code for last sequence
			if (i == seq_val_trimmed_size[0] - 1) {
				seq_start[old_sup_seq_idx[i] + 1] = i + 1;
				//check if last sequence is an append
				if (seq_trimmed[i * (single_seq_size[0] - 2) - 2] == ',') {
					//if true then previous sub database does not have any assemblages, so set assemble start[i] = seq_start[i]
					assemble_start[old_sup_seq_idx[i]] = i + 1;
				}
				// check if this sequence is an append
				if (seq_trimmed[(i + 1) * (single_seq_size[0] - 2) - 2] == ',') {
					//if true then last sub database has only appends, so set assemble start[i] = seq_start[i + 1]
					assemble_start[old_sup_seq_idx[i]] = i + 1;
				}
			}

			//default code:
			//new sub database starts when superior sequence index changes
			if (old_sup_seq_idx[i] > old_sup_seq_idx[i - 1]) {
				seq_start[old_sup_seq_idx[i]] = i;
				// check if previous sequence was an append
				if (seq_trimmed[i * (single_seq_size[0] - 2) - 2] == ',') {
					//if true then previous sub database does not have any assemblages, so set assemble start[i-1] = seq_start[i]
					assemble_start[old_sup_seq_idx[i] - 1] = i;
				}
				// check if this sequence is an assemblage
				if (seq_trimmed[(i + 1) * (single_seq_size[0] - 2) - 2] == '_') {
					//if true then this sub database has only assemblages, so set assemble start[i] = seq_start[i]
					assemble_start[old_sup_seq_idx[i]] = i;
				}
			}
			//check where assemblages start
			if (seq_trimmed[(i + 1) * (single_seq_size[0] - 2) - 2] == '_' && seq_trimmed[i * (single_seq_size[0] - 2) - 2] == ',') {
				assemble_start[old_sup_seq_idx[i]] = i;
			}
		}
	}
}


__global__ void generateNewSeq(char* seq, char* seq_trimmed, int* seq_val_trimmed, int* sup_seq_idx, int* old_sup_seq_idx, int* single_seq_size, unsigned int* append,
	unsigned int* assemble, unsigned int* last_element, int* seq_start, int* assemble_start, int* seq_val_idx, int* sup_database_size, int* new_row_idx_size, int* seq_val_trimmed_size) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int seq_idx = 0;
	int n_ap;
	int n_as;

	if (i < seq_val_trimmed_size[0]) {
		//Determine where given thread should write their sequences:
		//seq_idx += all new sequences from sub databases that come before sub database in which "i" sequence is in
		for (int j = 0; j < old_sup_seq_idx[i]; j++) {
			n_ap = assemble_start[j] - seq_start[j];
			n_as = seq_start[j + 1] - assemble_start[j];
			seq_idx += n_ap * n_ap + n_ap * (n_ap - 1) / 2 + n_ap * n_as + n_as * (n_as - 1) / 2;;
		}

		n_ap = assemble_start[old_sup_seq_idx[i]] - seq_start[old_sup_seq_idx[i]];
		n_as = seq_start[old_sup_seq_idx[i] + 1] - assemble_start[old_sup_seq_idx[i]];

		//seq_idx += all new sequences from database in which "i" sequence is in generated from append that come before "i" sequence
		for (int j = seq_start[old_sup_seq_idx[i]]; j < i && j < assemble_start[old_sup_seq_idx[i]]; j++) {
			seq_idx += 2 * n_ap - j - 1 + seq_start[old_sup_seq_idx[i]];
		}

		//seq_idx += all new sequences from database in which "i" sequence is in generated from assemblages that come before "i" sequence (if any)
		for (int j = assemble_start[old_sup_seq_idx[i]]; j < i; j++) {
			seq_idx += n_ap + n_as - j - 1 + assemble_start[old_sup_seq_idx[i]];
		}

		seq_val_idx[i] = seq_idx;

		//create new sequences:
		//create appends
		for (int j = seq_start[old_sup_seq_idx[i]]; j < assemble_start[old_sup_seq_idx[i]]; j++) {	
			for (int k = 0; k < single_seq_size[0] - 2; k++) {
				seq[seq_idx * single_seq_size[0] + k] = seq_trimmed[i * (single_seq_size[0] - 2) + k];
			}
			seq[(seq_idx + 1) * single_seq_size[0] - 2] = ',';
			seq[(seq_idx + 1) * single_seq_size[0] - 1] = seq_trimmed[(j + 1) * (single_seq_size[0] - 2) - 1];
			BIT_SET(append[i], seq_trimmed[(j + 1) * (single_seq_size[0] - 2) - 1] - 97);
			sup_seq_idx[seq_idx] = i;
			seq_idx++;
		}

		//create assemblages
		if (i < assemble_start[old_sup_seq_idx[i]]) { 
			for (int j = i + 1; j < assemble_start[old_sup_seq_idx[i]]; j++) {
				for (int k = 0; k < single_seq_size[0] - 2; k++) {
					seq[seq_idx * single_seq_size[0] + k] = seq_trimmed[i * (single_seq_size[0] - 2) + k];
				}
				seq[(seq_idx + 1) * single_seq_size[0] - 2] = '_';
				seq[(seq_idx + 1) * single_seq_size[0] - 1] = seq_trimmed[(j + 1) * (single_seq_size[0] - 2) - 1];
				BIT_SET(assemble[i], seq_trimmed[(j + 1) * (single_seq_size[0] - 2) - 1] - 97);
				sup_seq_idx[seq_idx] = i;
				seq_idx++;
			}
			new_row_idx_size[i + 1] = seq_val_trimmed[i] * (2 * n_ap - i - 1 + seq_start[old_sup_seq_idx[i]]);//???
		}
		else {
			for (int j = i + 1; j < seq_start[old_sup_seq_idx[i] + 1]; j++) {
				for (int k = 0; k < single_seq_size[0] - 2; k++) {
					seq[seq_idx * single_seq_size[0] + k] = seq_trimmed[i * (single_seq_size[0] - 2) + k];
				}
				seq[(seq_idx + 1) * single_seq_size[0] - 2] = '_';
				seq[(seq_idx + 1) * single_seq_size[0] - 1] = seq_trimmed[(j + 1) * (single_seq_size[0] - 2) - 1];
				BIT_SET(assemble[i], seq_trimmed[(j + 1) * (single_seq_size[0] - 2) - 1] - 97);
				sup_seq_idx[seq_idx] = i;
				seq_idx++;
			}
			new_row_idx_size[i + 1] = seq_val_trimmed[i] * (n_as + n_ap - i - 1 + assemble_start[old_sup_seq_idx[i]]);//???
		}

		//create last elements
		for (int j = single_seq_size[0] - 3; seq_trimmed[i * (single_seq_size[0] - 2) + j] != ',' && j > -1; j--) {
			if (seq_trimmed[i * (single_seq_size[0] - 2) + j] != '_') {
				BIT_SET(last_element[i], seq_trimmed[i * (single_seq_size[0] - 2) + j] - 97);
			}
		}
		
		sup_database_size[i] = seq_val_trimmed[i];
	}
}

int main(int argc, char** argv)
{
	auto start = std::chrono::high_resolution_clock::now();

	//enum for measuring exec time of different parts of program
	/*enum TimeMeasure
	{
		disk_to_RAM, RAM_to_VRAM, scan_db, prefix_sum, stream_compaction, VRAM_to_RAM, RAM_to_disk, other_GPU, write_results, init
	};
	trackTime(TimeMeasure::init);*/
	
	int h_unique_item_count; //value stating how many unique items are in database
	//check argc
	switch (argc) {
	case 4:
		h_unique_item_count = 26;
		break;
	case 5:
		h_unique_item_count = std::stoi(argv[4]);
		break;
	default:
		std::cout << "Incorrect number of input arguments\n";
		return 0;
	}

	float h_float_min_sup = std::stof(argv[3]); //minimum support

	if (h_float_min_sup > 1 || h_float_min_sup <= 0) {
		std::cout << "Incorrect minimum support value\n";
		return 0;
	}

	//host variables:
	std::ifstream file(argv[1]); //input database file
	std::ofstream output(argv[2]); //output sequences file
	std::string output_string; // string for writing to output file
	std::string h_database; //all database items in a single string
	std::vector<int> h_row_idx_start; //starting point of all rows in database
	std::vector<int> h_seq_val(h_unique_item_count); //frequency of sequences
	std::vector<char> h_seq(h_unique_item_count); //potential frequent sequences
	//std::vector<int> h_target_index; //DEBUG
	//std::vector<int> h_debug; //DEBUG
	std::vector<std::string> h_freq_seq; //found frequent sequences
	std::vector<int> h_seq_start; //index for first sequence in all sub databases, used to calculate number of appends and assemblages in sub databases
	std::vector<int> h_assemble_start;//index for first assemblage in all sub databases, used to calculate number of appends and assemblages in sub databases

	int h_database_size;//number of rows in database
	int h_new_database_size; //number of database rows in next iteration
	int h_seq_size;//number of potential frequent sequences
	int h_single_seq_size = 1;//single sequence size = 2n + 1, where n = number of algorithm iterations
	int h_trimmed_seq_size;//how many frequent sequences were found in given iteration
	int h_seq_start_size; 
	int h_min_sup; //minimum support expressed as number of rows in database
	

	//device variables:
	char* d_database;//all database items in a single string
	int* d_row_idx_start;//starting point of all rows in database
	int* d_seq_val;//frequency of sequences
	int* d_seq_val_trimmed;//frequency of sequences after trimming infrequent sequences
	int* d_new_row_idx_start;//starting point of all rows in database in next iteration
	int* d_database_size; //number of rows in database
	int* d_target_index; //variable for identifying which values of d_new_row_idx_start should be written to d_row_idx_start during stream compaction
	unsigned int* d_append; //appends that are being searched for in given datbase
	unsigned int* d_assemble; //assemblages thet are being searched for in given database
	unsigned int* d_last_element; //items that make up last element of prefix in given database
	int* d_sup_seq_idx; //index identifying superior database of given sequence
	int* d_old_sup_seq_idx; //index identifying superior database of given sequence from previous iteration
	int* d_sup_database_size; // number of rows that contained frequent sequences in previous iteration
	int* d_seq_val_idx; //index for identyfing which value in d_seq_val should be incremented after finding frequent sequence
	int* d_new_row_idx_size; //number of potential new database rows in next iteration
	int* d_adj_diff; //used to correct values in d_old_sup_seq_idx
	
	int* d_single_seq_size;//number of chars that form a single sequence = 2n + 1, where n = number of algorithm iterations
	char* d_seq;//potential frequent sequences
	char* d_seq_trimmed;//found frequent sequences
	int* d_seq_start;//index for first sequence in all sub databases, used to calculate number of appends and assemblages in sub databases
	int* d_assemble_start;//index for first assemblage in all sub databases, used to calculate number of appends and assemblages in sub databases
	int* d_seq_val_trimmed_size;//how many frequent sequences were found in given iteration
	int* d_new_database_size;//number of database rows in next iteration
	int* d_seq_size;

	//host variables for launching kernels:
	dim3 grid_size;
	dim3 block_size;
	const unsigned int max_block_size = 512;

	cudaError_t cudaStatus; //error container
	/*size_t h_free_memory;
	size_t h_total_memory;
	unsigned long long h_requied_memory;
	unsigned long long h_allocation_size;
	int h_partition_num;
	bool h_alloc_success;*/


	for (int i = 0; i < h_unique_item_count; i++) {
		h_seq[i] = 97 + i;
	}

	//open file with input database
	if (file.is_open()) {
		std::string line;
		int indexes_iterator = 0;

		h_row_idx_start.push_back(0);

		//load database
		while (std::getline(file, line)) {
			h_database += line;
			indexes_iterator += line.size();
			h_row_idx_start.push_back(indexes_iterator);
		}
		h_row_idx_start.pop_back();
		file.close();
	}
	else {
		std::cout << "Unable to open file";
		return 0;
	}

	h_database_size = h_row_idx_start.size();
	h_new_database_size = h_database_size * h_unique_item_count;
	h_min_sup = ceil(h_database_size * h_float_min_sup);

	//trackTime(TimeMeasure::disk_to_RAM);

	//allocate memory on GPU and copy data
	cudaStatus = cudaMalloc((void**)&d_database, sizeof(char) * h_database.size());
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_database, h_database.data(), sizeof(char) * h_database.size(), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_row_idx_start, sizeof(int) * h_row_idx_start.size());
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_row_idx_start, h_row_idx_start.data(), sizeof(int) * h_row_idx_start.size(), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_val, sizeof(int) * h_seq_val.size());
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_seq_val, 0, sizeof(int) * h_seq_val.size());
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq, sizeof(char) * h_seq.size());
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_seq, h_seq.data(), sizeof(char) * h_seq.size(), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_new_row_idx_start, sizeof(int) * h_unique_item_count * h_database_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_new_row_idx_start, 0, sizeof(int) * h_unique_item_count * h_database_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_database_size, sizeof(int));
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_database_size, &h_database_size, sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_size, sizeof(int));
	checkError(cudaStatus);

	//trackTime(TimeMeasure::RAM_to_VRAM);

	//determine block and grid size
	if (h_database_size > max_block_size) {
		grid_size = { (unsigned int)(h_database_size + max_block_size - 1) / max_block_size, 1, 1 };
		block_size = { max_block_size, 1, 1 };
	}
	else {
		grid_size = { 1, 1, 1 };
		block_size = { (unsigned int)h_database_size, 1, 1 };
	}

	//launch first database scan to find all length 1 sequences
	firstScan <<<grid_size, block_size >>> (d_database, d_row_idx_start, d_seq_val, d_new_row_idx_start, d_database_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus);

	//trackTime(TimeMeasure::scan_db);

	cudaStatus = cudaMalloc((void**)&d_target_index, sizeof(int) * h_unique_item_count * h_database_size);
	checkError(cudaStatus);

	cudaStatus = cudaMemcpyToSymbol(d_min_sup, &h_min_sup, sizeof(int));
	checkError(cudaStatus);

	cudaStatus = cudaMemcpyToSymbol(d_unique_item_count, &h_unique_item_count, sizeof(int));
	checkError(cudaStatus);

	//trackTime(TimeMeasure::RAM_to_VRAM);

	//check which rows didn't contain frequent sequence
	prepareVector <<<grid_size, block_size >>> (d_new_row_idx_start, d_target_index, d_seq_val, d_database_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus);

	//trackTime(TimeMeasure::other_GPU);

	//DEBUG
	/*h_target_index.resize(h_new_database_size);
	cudaStatus = cudaMemcpy(h_target_index.data(), d_new_row_idx_start, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);*/
	//DEBUG_END

	//Prefix sum with thrust
	thrust::device_ptr<int> d_thrust_ptr = thrust::device_pointer_cast(d_target_index);
	thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_new_database_size, d_thrust_ptr);

	//trackTime(TimeMeasure::prefix_sum);

	cudaStatus = cudaMemcpy(&h_database_size, d_target_index + (h_new_database_size - 1), sizeof(int), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	cudaStatus = cudaMemcpy(d_database_size, &h_new_database_size, sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaFree(d_row_idx_start);
	cudaStatus = cudaMalloc((void**)&d_row_idx_start, sizeof(int) * h_database_size);
	checkError(cudaStatus);

	//determine block and grid size
	if (h_new_database_size > max_block_size) {
		grid_size = { (unsigned int)(h_new_database_size + max_block_size - 1) / max_block_size, 1, 1 };
		block_size = { max_block_size, 1, 1 };
	}
	else {
		grid_size = { 1, 1, 1 };
		block_size = { (unsigned int)h_new_database_size, 1, 1 };
	}

	//Stream compaction to leave only rows with frequent sequences
	streamCompaction <<<grid_size, block_size >>> (d_new_row_idx_start, d_target_index, d_row_idx_start, d_database_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus);

	//trackTime(TimeMeasure::stream_compaction);

	cudaStatus = cudaMemcpy(d_database_size, d_target_index + (h_new_database_size - 1), sizeof(int), cudaMemcpyDeviceToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMemcpy(h_seq_val.data(), d_seq_val, sizeof(int) * h_seq_val.size(), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	//trackTime(TimeMeasure::VRAM_to_RAM);
	//DEBUG
	/*h_target_index.resize(h_database_size);
	cudaStatus = cudaMemcpy(h_target_index.data(), d_row_idx_start, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);*/
	//DEBUG_END

	//Free memory on device
	cudaFree(d_new_row_idx_start);
	cudaFree(d_target_index);

	//check which sequences to trim
	cudaStatus = cudaMalloc((void**)&d_target_index, sizeof(int) * h_seq_val.size());
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_size, sizeof(int));
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_seq_size, &h_unique_item_count, sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	//trackTime(TimeMeasure::RAM_to_VRAM);

	grid_size = { 1, 1, 1 };
	block_size = { (unsigned int)h_unique_item_count, 1, 1 };

	prepareSeq <<<grid_size, block_size >>> (d_seq_val, d_target_index, d_seq_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus);

	//Prefix sum with thrust
	d_thrust_ptr = thrust::device_pointer_cast(d_target_index);
	thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_seq_val.size(), d_thrust_ptr);

	cudaStatus = cudaMemcpy(&h_trimmed_seq_size, d_target_index + (h_seq_val.size() - 1), sizeof(int), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_val_trimmed, sizeof(int) * h_trimmed_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_trimmed, sizeof(char) * h_trimmed_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_single_seq_size, sizeof(int));
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_single_seq_size, &h_single_seq_size, sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_sup_seq_idx, sizeof(int) * h_unique_item_count);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_sup_seq_idx, 0, sizeof(int) * h_unique_item_count);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_old_sup_seq_idx, sizeof(int) * h_trimmed_seq_size);
	checkError(cudaStatus);

	//trim infrequent sequences
	trimSeq <<<grid_size, block_size >>> (d_seq, d_seq_val, d_seq_trimmed, d_seq_val_trimmed, d_target_index, d_sup_seq_idx, d_old_sup_seq_idx, d_single_seq_size, d_seq_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus);

	//trackTime(TimeMeasure::other_GPU);

	//copy results to CPU
	h_seq_val.resize(h_trimmed_seq_size);
	cudaStatus = cudaMemcpy(h_seq_val.data(), d_seq_val_trimmed, sizeof(int) * h_seq_val.size(), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	h_seq.resize(h_trimmed_seq_size);
	cudaStatus = cudaMemcpy(h_seq.data(), d_seq_trimmed, sizeof(char) * h_seq.size(), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	
	for (int i = 0; i < h_seq.size(); i++) {
		output_string += h_seq[i];
		output_string += ' ';
		output_string += std::to_string(h_seq_val[i]);
		output_string += '\n';
	}
	
	/*for (int i = 0; i < h_seq.size(); i++) {
		h_freq_seq.push_back({});
		h_freq_seq[i].push_back(h_seq[i]);
	}*/

	if (h_seq_val.size() == 0) {
		return 0;
	}

	//trackTime(TimeMeasure::VRAM_to_RAM);

	//print frequent sequences 
	/*for (int i = 0; i < std::size(h_seq_val); i++) {
		std::cout << h_freq_seq[i] << " " << h_seq_val[i] << "\n";
	}*/

	//write found sequences to output file
	if (output.is_open()) {
		output << output_string;
		/*for (int i = 0; i < std::size(h_seq_val); i++) {
			output << h_freq_seq[i] << " " << h_seq_val[i] << "\n";
		}*/
	}
	else {
		std::cout << "Unable to open output file";
		return 0;
	}

	//trackTime(TimeMeasure::RAM_to_disk);

	cudaFree(d_seq);
	cudaFree(d_seq_val);
	cudaFree(d_target_index);
	cudaFree(d_sup_seq_idx);

	//Prepare new sequences
	h_single_seq_size = h_single_seq_size + 2; //single sequence size = 2n + 1, where n = number of algorithm iterations
	h_seq_size = h_trimmed_seq_size * h_trimmed_seq_size + h_trimmed_seq_size * (h_trimmed_seq_size - 1) / 2;

	cudaStatus = cudaMalloc((void**)&d_seq, sizeof(char) * h_seq_size * h_single_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_val, sizeof(int) * h_seq_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_seq_val, 0, sizeof(int) * h_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_sup_seq_idx, sizeof(int) * h_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMemcpy(d_single_seq_size, &h_single_seq_size, sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_append, sizeof(unsigned int) * h_trimmed_seq_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_append, 0, sizeof(unsigned int) * h_trimmed_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_assemble, sizeof(unsigned int) * h_trimmed_seq_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_assemble, 0, sizeof(unsigned int) * h_trimmed_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_last_element, sizeof(unsigned int) * h_trimmed_seq_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_last_element, 0, sizeof(unsigned int) * h_trimmed_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_seq_val_trimmed_size, sizeof(int));
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_seq_val_trimmed_size, &h_trimmed_seq_size, sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_sup_database_size, sizeof(int) * h_trimmed_seq_size);
	checkError(cudaStatus); 

	cudaStatus = cudaMalloc((void**)&d_seq_val_idx, sizeof(int) * h_trimmed_seq_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_new_row_idx_size, sizeof(int) * (h_trimmed_seq_size + 1));
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_new_row_idx_size, 0, sizeof(int));
	checkError(cudaStatus);

	grid_size = { 1, 1, 1 };
	block_size = { (unsigned int)h_trimmed_seq_size, 1, 1 };

	firstGenerateNewSeq <<<grid_size, block_size >>> (d_seq, d_seq_trimmed, d_seq_val_trimmed, d_sup_seq_idx, d_single_seq_size, d_append, d_assemble, d_last_element, d_sup_database_size, d_seq_val_idx, d_new_row_idx_size, d_seq_val_trimmed_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	checkError(cudaStatus);

	cudaFree(d_seq_val_trimmed);
	cudaFree(d_seq_trimmed);

	//DB
	/*h_target_index.resize(h_trimmed_seq_size);
	cudaStatus = cudaMemcpy(h_target_index.data(), d_sup_database_size, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);*/
	//DBEND

	d_thrust_ptr = thrust::device_pointer_cast(d_new_row_idx_size);
	thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_trimmed_seq_size + 1, d_thrust_ptr);

	d_thrust_ptr = thrust::device_pointer_cast(d_sup_database_size);
	thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_trimmed_seq_size, d_thrust_ptr);

	cudaStatus = cudaMemcpy(&h_new_database_size, d_new_row_idx_size + h_trimmed_seq_size, sizeof(int), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	//DB
	//std::cout << "new database size " << h_new_database_size << '\n';
	//DBEND

	cudaStatus = cudaMalloc((void**)&d_new_database_size, sizeof(int));
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(d_new_database_size, d_new_row_idx_size + h_trimmed_seq_size, sizeof(int), cudaMemcpyDeviceToDevice);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_target_index, sizeof(int) * h_new_database_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_target_index, 0, sizeof(int) * h_new_database_size);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_new_row_idx_start, sizeof(int) * h_new_database_size);
	checkError(cudaStatus);
	cudaStatus = cudaMemset(d_new_row_idx_start, 0, sizeof(int) * h_new_database_size);
	checkError(cudaStatus);

	cudaFree(d_old_sup_seq_idx);

	//trackTime(TimeMeasure::other_GPU);

	//DEBUG
	/*h_seq.resize(h_seq_size * h_single_seq_size);
	cudaStatus = cudaMemcpy(h_seq.data(), d_seq, sizeof(char) * h_seq_size * h_single_seq_size, cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	h_target_index.resize(h_trimmed_seq_size);
	cudaStatus = cudaMemcpy(h_target_index.data(), d_seq_val_idx, sizeof(int) * h_trimmed_seq_size, cudaMemcpyDeviceToHost);
	checkError(cudaStatus);*/
	//DEBUG_END

	//repeat until no frequent sequences have been found
	while (h_seq_val.empty() == false) {

		//determine block and grid size
		if (h_database_size > max_block_size) {
			grid_size = { (unsigned int)(h_database_size + max_block_size - 1) / max_block_size, 1, 1 };
			block_size = { max_block_size, 1, 1 };
		}
		else {
			grid_size = { 1, 1, 1 };
			block_size = { (unsigned int)h_database_size, 1, 1 };
		}
		
		//DEBUG
		/*h_target_index.resize(h_new_database_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_target_index, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		//DEBUGEND

		//search database for new sequences 
		scanDatabase <<<grid_size, block_size >>> (d_database_size, d_sup_database_size, d_append, d_assemble, d_row_idx_start, d_database, d_seq_val, d_new_row_idx_start, d_seq_val_idx, d_last_element, d_new_row_idx_size, d_target_index, d_seq_val_trimmed_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);
		
		cudaFree(d_append);
		cudaFree(d_assemble);
		cudaFree(d_last_element);
		cudaFree(d_sup_database_size);
		cudaFree(d_seq_val_idx);
		cudaFree(d_new_row_idx_size);

		//trackTime(TimeMeasure::scan_db);

		//determine block and grid size
		if (h_new_database_size > max_block_size) {
			grid_size = { (unsigned int)(h_new_database_size + max_block_size - 1) / max_block_size, 1, 1 };
			block_size = { max_block_size, 1, 1 };
		}
		else {
			grid_size = { 1, 1, 1 };
			block_size = { (unsigned int)h_new_database_size, 1, 1 };
		}

		//check which rows didn't contain frequent sequence
		prepareVector2 <<<grid_size, block_size >>> (d_new_row_idx_start, d_target_index, d_seq_val, d_new_database_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		//DEBUG
		/*h_target_index.resize(h_new_database_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_target_index, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		//DEBUGEND

		//trackTime(TimeMeasure::other_GPU);

		//prefix sum
		d_thrust_ptr = thrust::device_pointer_cast(d_target_index);
		thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_new_database_size, d_thrust_ptr);

		//trackTime(TimeMeasure::prefix_sum);

		cudaStatus = cudaMemcpy(&h_database_size, d_target_index + (h_new_database_size - 1), sizeof(int), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		cudaStatus = cudaMemcpy(d_database_size, d_target_index + (h_new_database_size - 1), sizeof(int), cudaMemcpyDeviceToDevice);
		checkError(cudaStatus);

		//trackTime(TimeMeasure::VRAM_to_RAM);

		cudaFree(d_row_idx_start);

		//DEBUG
		//std::cout <<"database size: " << h_database_size << '\n';
		//DEBUGEND

		cudaStatus = cudaMalloc((void**)&d_row_idx_start, sizeof(int) * h_database_size); //HUGE MALLOC HERE
		checkError(cudaStatus);

		//create new database using rows that contained frequent sequences
		streamCompaction <<<grid_size, block_size >>> (d_new_row_idx_start, d_target_index, d_row_idx_start, d_new_database_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		cudaFree(d_new_row_idx_start);
		cudaFree(d_target_index);

		//trackTime(TimeMeasure::stream_compaction);
		//DEBUG
		/*h_seq_val.resize(h_seq_size);
		cudaStatus = cudaMemcpy(h_seq_val.data(), d_seq_val, sizeof(int) * h_seq_val.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		h_seq.resize(h_seq_size * h_single_seq_size);
		cudaStatus = cudaMemcpy(h_seq.data(), d_seq, sizeof(char) * h_seq.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		//DEBUGEND

		//Free memory on device
		//cudaFree(d_seq_val);
		

		cudaStatus = cudaMalloc((void**)&d_target_index, sizeof(int) * h_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMemcpy(d_seq_size, &h_seq_size, sizeof(int), cudaMemcpyHostToDevice);
		checkError(cudaStatus);

		//DEBUG
		/*std::cout << "h_seq_size: " << h_seq_size << '\n';*/
		//DEBUGEND

		//determine block and grid size
		if (h_seq_size > max_block_size) {
			grid_size = { (unsigned int)(h_seq_size + max_block_size - 1) / max_block_size, 1, 1 };
			block_size = { max_block_size, 1, 1 };
		}
		else {
			grid_size = { 1, 1, 1 };
			block_size = { (unsigned int)h_seq_size, 1, 1 };
		}

		//check which sequences to trim
		prepareSeq <<<grid_size, block_size >>> (d_seq_val, d_target_index, d_seq_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		//Prefix sum with thrust
		d_thrust_ptr = thrust::device_pointer_cast(d_target_index);
		thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_seq_size, d_thrust_ptr);

		//DEBUG
		/*h_target_index.resize(h_seq_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_target_index, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		std::cout << "d_target_index\n";
		for (int i = 0; i < h_target_index.size(); i++) {
			std::cout << h_target_index[i] << '\n';
		}
		std::cout << "d_target_index\n";*/
		//DEBUG_END

		cudaStatus = cudaMemcpy(&h_trimmed_seq_size, d_target_index + (h_seq_size - 1), sizeof(int), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		if (h_trimmed_seq_size == 0) { break; }

		cudaStatus = cudaMalloc((void**)&d_seq_val_trimmed, sizeof(int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_seq_trimmed, sizeof(char) * h_trimmed_seq_size * h_single_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_old_sup_seq_idx, sizeof(int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		//trim infrequent sequences
		trimSeq <<<grid_size, block_size >>> (d_seq, d_seq_val, d_seq_trimmed, d_seq_val_trimmed, d_target_index, d_sup_seq_idx, d_old_sup_seq_idx, d_single_seq_size, d_seq_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		//DEBUG
		/*h_debug.resize(h_trimmed_seq_size);
		cudaStatus = cudaMemcpy(h_debug.data(), d_old_sup_seq_idx, sizeof(int) * h_debug.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		std::cout << "d_old_sup_seq_idx\n";
		for (int i = 0; i < h_debug.size(); i++) {
			std::cout << h_debug[i] << '\n';
		}
		std::cout << "d_old_sup_seq_idx_end\n";*/
		//DEBUGEND

		//free memory
		cudaFree(d_seq_val);
		cudaFree(d_sup_seq_idx);
		cudaFree(d_seq);

		cudaStatus = cudaMalloc((void**)&d_adj_diff, sizeof(int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		//determine block and grid size
		if (h_trimmed_seq_size > max_block_size) {
			grid_size = { (unsigned int)(h_trimmed_seq_size + max_block_size - 1) / max_block_size, 1, 1 };
			block_size = { max_block_size, 1, 1 };
		}
		else {
			grid_size = { 1, 1, 1 };
			block_size = { (unsigned int)h_trimmed_seq_size, 1, 1 };
		}

		//correct values of d_old_sup_seq_idx so that difference between adjecent values is <= 1
		correctOldSupSeqIdx <<<grid_size, block_size >>> (d_old_sup_seq_idx, d_adj_diff);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		//DEBUG
		//h_target_index.resize(h_trimmed_seq_size);
		//cudaStatus = cudaMemcpy(h_target_index.data(), d_adj_diff, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		//checkError(cudaStatus);

		//h_debug.resize(h_trimmed_seq_size);
		//cudaStatus = cudaMemcpy(h_debug.data(), d_old_sup_seq_idx, sizeof(int) * h_debug.size(), cudaMemcpyDeviceToHost);
		//checkError(cudaStatus);
		//
		//for (int i = 0; i < h_target_index.size(); i++) {
		//	std::cout /*<< h_target_index[i]*/ << ' ' << h_debug[i] << '\n';
		//}
		//DEBUGEND

		d_thrust_ptr = thrust::device_pointer_cast(d_adj_diff);
		thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_trimmed_seq_size, d_thrust_ptr);
		

		thrust::device_ptr<int> d_thrust_ptr2 = thrust::device_pointer_cast(d_old_sup_seq_idx);
		thrust::transform(d_thrust_ptr2, d_thrust_ptr2 + h_trimmed_seq_size, d_thrust_ptr, d_thrust_ptr2, thrust::minus<int>());

		//DEBUG
		/*h_target_index.resize(h_trimmed_seq_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_adj_diff, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		////DEBUGEND

		////DEBUG
		/*h_debug.resize(h_trimmed_seq_size);
		cudaStatus = cudaMemcpy(h_debug.data(), d_old_sup_seq_idx, sizeof(int) * h_debug.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		////DEBUGEND

		////DEBUG
		//for (int i = 0; i < h_target_index.size(); i++) {
		//	std::cout << h_target_index[i] << ' ' << h_debug[i] << '\n';
		//}
		//DEBUGEND

		cudaFree(d_adj_diff);
		//trackTime(TimeMeasure::other_GPU);
	
		//DEBUG
		/*h_target_index.resize(h_trimmed_seq_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_old_sup_seq_idx, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		//DEBUGEND

		//copy results to CPU
		cudaStatus = cudaMemcpy(&h_seq_start_size, d_old_sup_seq_idx + (h_trimmed_seq_size - 1), sizeof(int), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);
		h_seq_start_size++;

		//DEBUG
		//std::cout << "h_seq_start_size: " << h_seq_start_size << '\n';
		//DEBUGEND

		h_seq_val.resize(h_trimmed_seq_size);
		cudaStatus = cudaMemcpy(h_seq_val.data(), d_seq_val_trimmed, sizeof(int) * h_seq_val.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		h_seq.resize(h_trimmed_seq_size * h_single_seq_size);
		cudaStatus = cudaMemcpy(h_seq.data(), d_seq_trimmed, sizeof(char) * h_seq.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		//trackTime(TimeMeasure::VRAM_to_RAM);

		output_string.clear();
		for (int i = 0; i < h_seq_val.size(); i++) {
			for (int j = 0; j < h_single_seq_size; j++) {
				output_string += h_seq[i * h_single_seq_size + j];
			}
			output_string += ' ';
			output_string += std::to_string(h_seq_val[i]);
			output_string += "\n";
		}

		/*h_freq_seq.resize(0);
		for (int i = 0; i < h_seq_val.size(); i++) {
			h_freq_seq.push_back({});
			for (int j = 0; j < h_single_seq_size; j++) {
				h_freq_seq[i].push_back(h_seq[i * h_single_seq_size + j]);
			}
		}*/

		//print frequent sequences //DEBUG
		/*for (int i = 0; i < std::size(h_seq_val); i++) {
			std::cout << h_freq_seq[i] << " " << h_seq_val[i] << "\n";
		}*/

		//write found sequences to output file
		output << output_string;
		/*for (int i = 0; i < std::size(h_seq_val); i++) {
			output << h_freq_seq[i] << " " << h_seq_val[i] << "\n";
		}*/

		/*for (int i = 0; i < std::size(h_seq_val); i++) {
			for (int j = 0; j < h_single_seq_size; j++) {
				output << h_seq[i * h_single_seq_size + j];
			}
			output << " " << h_seq_val[i] << "\n";
		}*/

		//trackTime(TimeMeasure::RAM_to_disk);

		//Prepare new sequences
		h_single_seq_size += 2; //single sequence size = 2n + 1, where n = number of algorithm iterations

		cudaStatus = cudaMalloc((void**)&d_seq_start, sizeof(int) * (h_seq_start_size + 1));
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_assemble_start, sizeof(int) * h_seq_start_size);
		checkError(cudaStatus);

		cudaStatus = cudaMemcpy(d_single_seq_size, &h_single_seq_size, sizeof(int), cudaMemcpyHostToDevice);
		checkError(cudaStatus);

		cudaStatus = cudaMemcpy(d_seq_val_trimmed_size, &h_trimmed_seq_size, sizeof(int), cudaMemcpyHostToDevice);
		checkError(cudaStatus);

		calculateNewSeqPos <<<grid_size, block_size >>> (d_seq_trimmed, d_seq_start, d_assemble_start, d_old_sup_seq_idx, d_single_seq_size, d_seq_val_trimmed_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		//DEBUG
		/*h_target_index.resize(h_trimmed_seq_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_old_sup_seq_idx, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);*/
		//DEBUGEND

		//Copy data to CPU
		h_seq_start.resize(h_seq_start_size + 1);
		cudaStatus = cudaMemcpy(h_seq_start.data(), d_seq_start, sizeof(int) * h_seq_start.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		h_assemble_start.resize(h_seq_start_size);
		cudaStatus = cudaMemcpy(h_assemble_start.data(), d_assemble_start, sizeof(int) * h_assemble_start.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		//DEBUG
		/*for (int i = 0; i < h_assemble_start.size(); i++) {
			std::cout << h_seq_start[i] << ' ' << h_assemble_start[i] << '\n';
		}
		std::cout << h_seq_start.back() << '\n';*/
		//DEBUGEND

		//calculate how much space needs to be allocated on GPU for new sequences 
		h_seq_size = 0;
		for (int i = 0; i < h_assemble_start.size(); i++) {
			int n_ap = h_assemble_start[i] - h_seq_start[i]; //number of appends in sub database
			int n_as = h_seq_start[i + 1] - h_assemble_start[i];//number of asemblages in sub database
			h_seq_size += n_ap * n_ap + n_ap * (n_ap - 1) / 2 + n_ap * n_as + n_as * (n_as - 1) / 2;
		}

		cudaStatus = cudaMalloc((void**)&d_seq, sizeof(char) * h_seq_size * h_single_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_seq_val, sizeof(int) * h_seq_size);
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_seq_val, 0, sizeof(int) * h_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_sup_seq_idx, sizeof(int) * h_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_append, sizeof(unsigned int) * h_trimmed_seq_size);
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_append, 0, sizeof(unsigned int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_assemble, sizeof(unsigned int) * h_trimmed_seq_size);
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_assemble, 0, sizeof(unsigned int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_last_element, sizeof(unsigned int) * h_trimmed_seq_size);
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_last_element, 0, sizeof(unsigned int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_seq_val_idx, sizeof(int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_sup_database_size, sizeof(int) * h_trimmed_seq_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_new_row_idx_size, sizeof(int) * (h_trimmed_seq_size + 1));
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_new_row_idx_size, 0, sizeof(int));
		checkError(cudaStatus);

		generateNewSeq <<<grid_size, block_size >>> (d_seq, d_seq_trimmed, d_seq_val_trimmed, d_sup_seq_idx, d_old_sup_seq_idx, d_single_seq_size, d_append,
			d_assemble, d_last_element, d_seq_start, d_assemble_start, d_seq_val_idx, d_sup_database_size, d_new_row_idx_size, d_seq_val_trimmed_size);

		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		checkError(cudaStatus);

		cudaFree(d_seq_trimmed);
		cudaFree(d_seq_start);
		cudaFree(d_assemble_start);
		cudaFree(d_old_sup_seq_idx);
		cudaFree(d_seq_val_trimmed);

		//DEBUG
		/*h_target_index.resize(h_seq_size);
		cudaStatus = cudaMemcpy(h_target_index.data(), d_sup_seq_idx, sizeof(int) * h_target_index.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);
		
		h_seq.resize(h_seq_size * h_single_seq_size);
		cudaStatus = cudaMemcpy(h_seq.data(), d_seq, sizeof(char) * h_seq.size(), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		for (int i = 0; i < h_target_index.size(); i++) {
			std::cout << h_target_index[i] << '\n';
		}

		std::cout << '\n';
		*/
		//DEBUG_END

		d_thrust_ptr = thrust::device_pointer_cast(d_new_row_idx_size);
		thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_trimmed_seq_size + 1, d_thrust_ptr);

		d_thrust_ptr = thrust::device_pointer_cast(d_sup_database_size);
		thrust::inclusive_scan(d_thrust_ptr, d_thrust_ptr + h_trimmed_seq_size, d_thrust_ptr);

		cudaStatus = cudaMemcpy(&h_new_database_size, d_new_row_idx_size + h_trimmed_seq_size, sizeof(int), cudaMemcpyDeviceToHost);
		checkError(cudaStatus);

		if (h_new_database_size == 0) { break; }

		//std::cout << "new database size "<< h_new_database_size << '\n';

		cudaStatus = cudaMemcpy(d_new_database_size, d_new_row_idx_size + h_trimmed_seq_size, sizeof(int), cudaMemcpyDeviceToDevice);
		checkError(cudaStatus);

		//Check available memory
		/*cudaMemGetInfo(&h_free_memory, &h_total_memory);
		std::cout << "free: " << h_free_memory << " total: " << h_total_memory << '\n';*/
		
		//h_requied_memory = 2 * h_new_database_size * sizeof(int);
		//h_partition_num = h_requied_memory / h_free_memory + 1;//+1 - round up
		//h_allocation_size = h_requied_memory / (h_partition_num * 2);

		//h_alloc_success = false;
		//while (h_alloc_success == false) {
		//	cudaStatus = cudaMalloc((void**)&d_target_index, h_allocation_size);
		//	if (cudaStatus == cudaSuccess) {
		//		cudaStatus = cudaMalloc((void**)&d_new_row_idx_start, h_allocation_size);
		//		if (cudaStatus == cudaSuccess) {
		//			h_alloc_success = true;
		//		}
		//		else{
		//			h_allocation_size -= 100 * 1024 * 1024; //reduce allocation size by 100 MB until succesful
		//			cudaFree(d_target_index);
		//			if (h_allocation_size < 100 * 1024 * 1024) {
		//				break;
		//			}
		//		}
		//	}
		//}
		//
		//if (h_alloc_success == false) {
		//	std::cout << "Error: not enough memory\n";
		//	break;
		//}

		//h_partition_num = h_requied_memory / (h_allocation_size * 2); // number of partitions

		//std::cout << h_partition_num << '\n';

		cudaStatus = cudaMalloc((void**)&d_target_index, sizeof(int) * h_new_database_size);
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_target_index, 0, sizeof(int) * h_new_database_size);
		checkError(cudaStatus);

		cudaStatus = cudaMalloc((void**)&d_new_row_idx_start, sizeof(int) * h_new_database_size);
		checkError(cudaStatus);
		cudaStatus = cudaMemset(d_new_row_idx_start, 0, sizeof(int) * h_new_database_size);
		checkError(cudaStatus);

		//trackTime(TimeMeasure::other_GPU);

		//Check available memory
		/*cudaMemGetInfo(&h_free_memory, &h_total_memory);
		std::cout << "free: " << h_free_memory << " total: " << h_total_memory << '\n';*/
	}

	cudaFree(d_single_seq_size);
	cudaFree(d_seq_val_trimmed_size);
	cudaFree(d_new_database_size);
	cudaFree(d_database);
	cudaFree(d_database_size);

	//trackTime(TimeMeasure::write_results);
	// stop timer
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds\n";
}