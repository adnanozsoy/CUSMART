#ifndef WRAPPER_LIST_H
#define WRAPPER_LIST_H

#ifdef __cplusplus
extern "C" {
#endif

#include "util/parameters.h"

struct wrapper_info
{
	const char* name;
	const char* tag;
	search_info (*search)(struct search_parameters);
};

extern search_info brute_force_wrapper(search_parameters);
extern search_info brute_force_block_wrapper(search_parameters);
extern search_info brute_force_block_shared_wrapper(search_parameters);
extern search_info morris_pratt_wrapper(search_parameters);
extern search_info knuth_morris_pratt_wrapper(search_parameters);
extern search_info simon_wrapper(search_parameters);
extern search_info apostolico_giancarlo_wrapper(search_parameters);
extern search_info quicksearch_wrapper(search_parameters);
extern search_info apostolico_crochemore_wrapper(search_parameters);
extern search_info colussi_wrapper(search_parameters);
extern search_info galil_giancarlo_wrapper(search_parameters);
extern search_info forward_dawg_wrapper(search_parameters);
extern search_info double_forward_dawg_wrapper(search_parameters);
extern search_info backward_nondeterministic_dawg_wrapper(search_parameters);
extern search_info reverse_colussi_wrapper(search_parameters);
extern search_info shift_vector_matching_wrapper(search_parameters);
extern search_info two_way_nondeterministic_dawg_wrapper(search_parameters);
extern search_info long_backward_nondeterministic_dawg_wrapper(search_parameters);
extern search_info simplified_backward_nondeterministic_dawg_unrolled_wrapper(search_parameters);
extern search_info forward_nondeterministic_dawg_wrapper(search_parameters);
extern search_info backward_fast_search_wrapper(search_parameters);
extern search_info backward_nondeterministic_dawg_qgram_wrapper(search_parameters);
extern search_info simplified_backward_nondeterministic_dawg_horspool_shift(search_parameters);
extern search_info bitparallel_wide_window_wrapper(search_parameters);
extern search_info forward_fast_search_wrapper(search_parameters);
extern search_info horspool_with_bndm_wrapper(search_parameters);
extern search_info fast_average_shift_optimal_or_wrapper(search_parameters);
extern search_info average_shift_optimal_or_wrapper(search_parameters);
extern search_info improved_linear_dawg1_wrapper(search_parameters);
extern search_info improved_linear_dawg2_wrapper(search_parameters);
extern search_info bit_parallel_length_invariant_matcher_wrapper(search_parameters);
extern search_info forward_simplified_backward_nondeterministic_dawg_matching_wrapper(search_parameters);
extern search_info backward_nondeterministic_dawg_qgram4_wrapper(search_parameters);
extern search_info simplified_backward_nondeterministic_dawg_qgram_wrapper(search_parameters);
extern search_info forward_simplified_bndm_qgram_schar_wrapper(search_parameters);
extern search_info brute_force_stream_wrapper(search_parameters);



extern search_info deterministic_finite_automaton_wrapper(search_parameters);
extern search_info high_deterministic_finite_automaton_wrapper(search_parameters);
extern search_info boyer_moore_wrapper(search_parameters);
extern search_info karp_rabin_wrapper(search_parameters);
extern search_info optimal_mismatch_wrapper(search_parameters);
extern search_info tuned_boyer_moore_wrapper(search_parameters);
extern search_info two_way_wrapper(search_parameters);
extern search_info string_matching_ordered_alphabet_wrapper(search_parameters);
extern search_info reverse_factor_wrapper(search_parameters);
extern search_info turbo_boyer_moore_wrapper(search_parameters);
extern search_info berry_ravindran_wrapper(search_parameters);
extern search_info fast_search_wrapper(search_parameters);
extern search_info backward_oracle_matching_wrapper(search_parameters);
extern search_info simplified_backward_nondeterministic_dawg_matching_wrapper(search_parameters);


extern search_info horspool_wrapper(search_parameters);
extern search_info zhu_takaoka_wrapper(search_parameters);
extern search_info maximal_shift_wrapper(search_parameters); 
extern search_info smith_wrapper(search_parameters);
extern search_info raita_wrapper(search_parameters);
extern search_info so_wrapper(search_parameters);
extern search_info sa_wrapper(search_parameters);
extern search_info nsn_wrapper(search_parameters);
extern search_info skip_search_wrapper(search_parameters);
extern search_info kmpskip_wrapper(search_parameters);
extern search_info ssabs_wrapper(search_parameters);
extern search_info tailed_substring_wrapper(search_parameters);
extern search_info wide_window_wrapper(search_parameters);
extern search_info linear_dawg_matching_wrapper(search_parameters);
extern search_info tvsbs_wrapper(search_parameters);
extern search_info fjs_wrapper(search_parameters);
extern search_info two_sliding_window_wrapper(search_parameters);
extern search_info pbmh_wrapper(search_parameters);
extern search_info turbo_reverse_factor_wrapper(search_parameters);
extern search_info ebom_wrapper(search_parameters);
extern search_info fbom_wrapper(search_parameters);
extern search_info sebom_wrapper(search_parameters);
extern search_info sfbom_wrapper(search_parameters);
extern search_info sabp_wrapper(search_parameters);
extern search_info bndm_extended_shift_wrapper(search_parameters);
extern search_info backward_snr_dawg_matching_wrapper(search_parameters);
extern search_info factorized_backward_nondeterministic_dawg_matching_wrapper(search_parameters);
extern search_info factorized_shift_and_wrapper(search_parameters);
extern search_info hash3_wrapper(search_parameters);

static struct wrapper_info wrapper_list[] = {
    {
        "Bruteforce",
        "bf",
        brute_force_wrapper,
    },
    {
        "Bruteforce block",
        "bfb",
        brute_force_block_wrapper,
    },
    {
        "Bruteforce block shared",
        "bfbs",
        brute_force_block_shared_wrapper,
    },
    {
        "Morris Pratt",
        "mp",
        morris_pratt_wrapper,
    },
    {
        "Knuth Morris Pratt",
        "kmp",
        knuth_morris_pratt_wrapper,
    },
    {
        "Simon",
        "simon",
        simon_wrapper,
    },  
    {
        "Det. Finite Automaton",
        "dfa",
        deterministic_finite_automaton_wrapper,
    },  
    {
        "High Finite Automaton",
        "dfah",
        high_deterministic_finite_automaton_wrapper,
    },
    {
        "Boyer Moore",
        "bm",
        boyer_moore_wrapper,
    },
    {
        "Apostolico Giancarlo",
        "ag",
        apostolico_giancarlo_wrapper,
    },
    {
        "Horspool",
        "hor",
        horspool_wrapper,
    },
    {
        "Zhu Takaoka",
        "zt",
        zhu_takaoka_wrapper,
    },  
    {
        "Karp Rabin",
        "kr",
        karp_rabin_wrapper,
    },
    {
        "Quicksearch",
        "qs",
        quicksearch_wrapper,
    },  
    {
        "Optimal Mismatch",
        "om",
        optimal_mismatch_wrapper,
    },
    {
        "Maximal Shift",
        "ms",
        maximal_shift_wrapper,
    },
    {
        "Smith",
        "smith",
        smith_wrapper,
    },
    {
        "Apostolico Crochemore",
        "ac",
        apostolico_crochemore_wrapper,
    },
    {
        "Raita",
        "raita",
        raita_wrapper,
    },
    {
        "Tuned Boyer Moore",
        "tunedbm",
        tuned_boyer_moore_wrapper,
    },
    {
        "Shift Or",
        "so",
        so_wrapper,
    },
    {
        "Shift And",
        "sa",
        sa_wrapper,
    },
    {
        "Not So Naive",
        "nsn",
        nsn_wrapper,
    },
    {
        "Colussi",
        "col",
        colussi_wrapper,
    },    
    {
        "Two Way",
        "tw",
        two_way_wrapper,
    },
    {
        "SM Ordered Alphabet",
        "smoa",
        string_matching_ordered_alphabet_wrapper,
    },
    {
        "Galil Giancarlo",
        "gg",
        galil_giancarlo_wrapper,
    },
    {
        "Reverse Factor",
        "rf",
        reverse_factor_wrapper,
    },
    {
        "Skip Search",
        "skip",
        skip_search_wrapper,
    },
    {
        "Turbo Boyer Moore",
        "tbm",
        turbo_boyer_moore_wrapper,
    },
    {
        "KMP Skip Search",
        "kmpskip",
        kmpskip_wrapper,
    },
    {
        "Berry Ravindran ",
        "br",
        berry_ravindran_wrapper,
    },
    {
        "Fast Search ",
        "fs",
        fast_search_wrapper,
    },
    {
        "Forward DAWG Matching",
        "fdm",
        forward_dawg_wrapper,
    },
    /*{
        "Double Forward DAWG Matching",
        "dfdm",
        double_forward_dawg_wrapper,
	},*/
    {
        "Backward nondet. DAWG Matching",
        "bndm",
        backward_nondeterministic_dawg_wrapper,
    },
    {
        "SSABS Algorithm",
        "ssabs",
        ssabs_wrapper,
    },
    {
        "Tailed Substring",
        "ts",
        tailed_substring_wrapper,
    },
    {
        "Backward Oracle Matching",
        "bom",
        backward_oracle_matching_wrapper,
    },
    {
        "Wide Window",
        "ww",
        wide_window_wrapper,
    },
    {
        "Reverse Colussi",
        "rcol",
        reverse_colussi_wrapper,
    },
    /*{
        "Shift Vector Matching",
        "svm",
        shift_vector_matching_wrapper,
	},*/
    {
        "Linear DAWG Matching",
        "ldm",
        linear_dawg_matching_wrapper,
    },
    {
        "Two-way nondet. DAWG Matching",
        "tndm",
        two_way_nondeterministic_dawg_wrapper,
    },
    {
        "TVSBS",
        "tvsbs",
        tvsbs_wrapper,
    },
    {
        "Simplified BNDM",
        "sbndm",
        simplified_backward_nondeterministic_dawg_matching_wrapper,
    },
    {
        "Long BNDM",
        "lbndm",
        long_backward_nondeterministic_dawg_wrapper,
    },
    {
        "Franek Jennings Smyth",
        "fjs",
        fjs_wrapper,
    },
    {
        "Simplified BNDM /w loop unrolling",
        "sbndm2",
        simplified_backward_nondeterministic_dawg_unrolled_wrapper,
    },
    {
        "Forward nondet. DAWG Matching",
        "fndm",
        forward_nondeterministic_dawg_wrapper,
    },
    {
        "Two Sliding Window",
        "tsw",
        two_sliding_window_wrapper,
    },
    {
        "Backward Fast Search",
        "bfs",
        backward_fast_search_wrapper,
    },
    {
        "BNDM with loop unrolling",
        "bndmq2",
        backward_nondeterministic_dawg_qgram_wrapper,
    },
    {
        "Simplified BNDM with Horspool Shift",
        "sbndm_bmh",
        simplified_backward_nondeterministic_dawg_matching_wrapper,
    },
    {
        "Boyer Moore Horspool using Probabilities",
        "pbmh",
        pbmh_wrapper,
    },
    {
        "Turbo Reverse Factor",
        "trf",
        turbo_reverse_factor_wrapper,
    },
    {
        "Bitparallel Wide Window",
        "bww",
    	bitparallel_wide_window_wrapper,
    },
    {
        "Extended Backward Oracle Matching",
        "ebom",
        ebom_wrapper,
    },
    {
        "Forward Backward Oracle Matching",
    	"fbom",
        fbom_wrapper,
    },
    {
        "Simplified Extended Backward Oracle Matching",
        "sebom",
        sebom_wrapper,
    },
    {
        "Simplified Forward Backward Oracle Matching",
        "sfbom",
        sfbom_wrapper,
    },
    {
        "Forward Fast Search",
        "ffs",
        forward_fast_search_wrapper,
    },
    {
        "Horspool with BNDM test",
        "bmh_sbndm",
        horspool_with_bndm_wrapper,
    },
    {
        "Fast Average Optimal Shift Or",
        "faoso2",
        fast_average_shift_optimal_or_wrapper,
    },
    {
        "Average Optimal Shift Or",
        "aoso2",
        average_shift_optimal_or_wrapper,
    },
    {
        "Improved Linear DAWG Matching 1",
        "ildm1",
        improved_linear_dawg1_wrapper,
    },
    {
        "Improved Linear DAWG Matching 2",
        "ildm2",
        improved_linear_dawg2_wrapper,
    },
    {
        "Small Alphabet Bit Parallel",
        "sabp",
        sabp_wrapper,
    },
    {
        "BNDM with Extended Shift",
        "bxs",
        bndm_extended_shift_wrapper,
    },
    {
        "Backward SNR DAWG Matching",
        "bsdm",
        backward_snr_dawg_matching_wrapper,
    },
    {
        "Factorized BNDM",
        "kbndm",
        factorized_backward_nondeterministic_dawg_matching_wrapper,
    },
    {
        "Factorized Shift And",
        "ksa",
        factorized_shift_and_wrapper,
    },
    {
        "Wu Manber for Single Pattern Matching",
        "hash3",
        hash3_wrapper,
    },
    {
        "Bit Parallel Length Invariant Matcher",
        "blim",
        bit_parallel_length_invariant_matcher_wrapper,
    },
    {
        "Forward Simplified BNDM",
        "fsbndm",
        forward_simplified_backward_nondeterministic_dawg_matching_wrapper,
    },
    {
        "Backward Nondet. DAWG Matching with q-grams",
        "bndmq4",
        backward_nondeterministic_dawg_qgram4_wrapper,
    },
    {
        "Simplified BNDM with q-grams",
        "sbndmq2",
        simplified_backward_nondeterministic_dawg_qgram_wrapper,
    },
    {
        "Forward SBNDM with q-grams & s-f characters",
        "fsbndmq20",
        forward_simplified_bndm_qgram_schar_wrapper,
    },
    {
        "Bruteforce Stream",
        "bfst",
        brute_force_stream_wrapper,
    },

};

static int wrapper_list_len = sizeof(wrapper_list)/sizeof(wrapper_list[0]);


#ifdef __cplusplus
}
#endif
#endif
