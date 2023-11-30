#ifndef ALGORITHM_LIST_H
#define ALGORITHM_LIST_H

#ifdef __cplusplus
extern "C" {
#endif

#include "util/parameters.h"

extern void bf(search_parameters);
extern void mp(search_parameters);
extern void kmp(search_parameters);
extern void simon(search_parameters);
extern void bm(search_parameters);
extern void dfa(search_parameters);
extern void ag(search_parameters);
extern void hor(search_parameters);
extern void zt(search_parameters);
extern void kr(search_parameters);
extern void qs(search_parameters);
extern void om(search_parameters);
extern void ms(search_parameters); 
extern void smith(search_parameters);
extern void ac(search_parameters); 
extern void raita(search_parameters);
extern void tuned_bm(search_parameters);
extern void so(search_parameters);
extern void sa(search_parameters);
extern void nsn(search_parameters);
extern void col(search_parameters); 
extern void tw(search_parameters);
extern void smoa(search_parameters);
extern void gg(search_parameters);
extern void rf(search_parameters);
extern void skip(search_parameters);
extern void tbm(search_parameters);
extern void kmpskip(search_parameters);
extern void br(search_parameters);
extern void fs(search_parameters);
extern void fdm(search_parameters);
extern void dfdm(search_parameters);
extern void bndm(search_parameters);
extern void ssabs(search_parameters);
extern void ts(search_parameters);
extern void bom(search_parameters);
// extern void akc(search_parameters);
extern void ww(search_parameters);
extern void rcolussi(search_parameters);
extern void svm(search_parameters);
extern void ldm(search_parameters);
// extern void sbndm(search_parameters);
extern void tndm(search_parameters);
extern void tvsbs(search_parameters);
extern void sbndm(search_parameters);
extern void lbndm(search_parameters);
extern void fjs(search_parameters);
extern void sbndm2(search_parameters);
extern void fndm(search_parameters);
extern void tsw(search_parameters);
extern void bfs(search_parameters);
extern void bndmq2(search_parameters);
extern void sbndm_bmh(search_parameters);
extern void pbmh(search_parameters);
extern void trf(search_parameters);
extern void bww(search_parameters);
extern void ebom(search_parameters);
extern void fbom(search_parameters);
extern void sebom(search_parameters);
extern void sfbom(search_parameters);
extern void ffs0(search_parameters);
extern void bmh_sbndm(search_parameters);
extern void faoso2(search_parameters);
extern void aoso2(search_parameters);
extern void ildm1(search_parameters);
extern void ildm2(search_parameters);
extern void sabp(search_parameters);
extern void bxs(search_parameters);
extern void bsdm(search_parameters);
extern void kbndm(search_parameters);
extern void ksa(search_parameters);
extern void hash3(search_parameters);
extern void blim(search_parameters);
extern void fsbndm(search_parameters);
extern void bndmq4(search_parameters);
extern void sbndmq2(search_parameters);
extern void fsbndmq20(search_parameters);

struct algorithm_info
{
    const char* name;
    const char* tag;
    void (*search)(struct search_parameters);
};


struct algorithm_info algorithm_list[] = {  
    {
        "Bruteforce",
        "bf",
        bf,
    },
    {
        "Morris Pratt",
        "mp",
        mp,
    },
    {
        "Knuth Morris Pratt",
        "kmp",
        kmp,
    },
    {
        "Simon",
        "simon",
        simon,
    },
    {
        "Det. Finite Automaton",
        "dfa",
        dfa,
    },
    {
        "Boyer Moore",
        "bm",
        bm,
    },
    {
        "Apostolico Giancarlo",
        "ag",
        ag,
    },
    {
        "Horspool",
        "hor",
        hor,
    },
    {
        "Zhu Takaoka",
        "zt",
        zt,
    },  
    {
        "Karp Rabin",
        "kr",
        kr,
    },
    {
        "Quicksearch",
        "qs",
        qs,
    },
    {
        "Optimal Mismatch",
        "om",
        om,
    },
    {
        "Maximal Shift",
        "ms",
        ms,
    },
    {
        "Smith",
        "smith",
        smith,
    },
    {
        "Apostolico-Crochemore",
        "ac",
        ac,
    },
    {
        "Raita",
        "raita",
        raita,
    },
    {
        "Tuned Boyer Moore",
        "tunedbm",
        tuned_bm,
    },
    {
        "Shift Or",
        "so",
        so,
    },
    {
        "Shift And",
        "sa",
        sa,
    },
    {
        "Not So Naive",
        "nsn",
        nsn,
    },
    {
        "Colussi",
        "col",
        col,
    },    
    {
        "Two Way",
        "tw",
        tw,
    },
    {
        "Ordered Alphabet",
        "smoa",
        smoa,
    },
    {
        "Galil Giancarlo",
        "gg",
        gg,
    },
    {
        "Reverse Factor",
        "rf",
        rf,
    },
    {
        "Skip Search",
        "skip",
        skip,
    },
    {
        "Turbo Boyer Moore",
        "tbm",
        tbm,
    },
    {
        "KMP Skip Search",
        "kmpskip",
        kmpskip,
    },
    {
        "Berry Ravindran",
        "br",
        br,
    },
    {
        "Fast Search",
        "fs",
        fs,
    },
    {
        "Forward DAWG Matching",
        "fdm",
        fdm,
    },
    {
        "Double Forward DAWG Matching",
        "dfdm",
        dfdm,
    },
    {
        "Backward nondet. DAWG Matching",
        "bndm",
        bndm,
    },
    {
        "SSABS Algorithm",
        "ssabs",
        ssabs,
    },
    {
        "Tailed Substring",
        "ts",
        ts,
    },
    {
        "Backward Oracle Matching",
        "bom",
        bom,
    },
    // {
    //     "ahmed kaykobad chowdhury",
    //     "akc",
    //     akc,
    // },
    {
        "Wide Window",
        "ww",
        ww,
    },
    {
        "Reverse Colussi",
        "rcol",
        rcolussi,
    },
    {
        "Shift Vector Matching",
        "svm",
        svm,
    },
    {
        "Linear DAWG Matching",
        "ldm",
        ldm,
    },
    {
        "Two-way nondet. DAWG Matching",
        "tndm",
        tndm,
    },
    {
        "TVSBS",
        "tvsbs",
        tvsbs,
    },
    {
        "Simplified BNDM",
        "sbndm",
        sbndm,
    },
    {
        "Long BNDM",
        "lbndm",
        lbndm,
    },
    {
        "Franek Jennings Smyth",
        "fjs",
        fjs,
    },
    {
        "Simplified BNDM /w loop unrolling",
        "sbndm2",
        sbndm2,
    },
    {
        "Forward  nondet. DAWG Matching",
        "fndm",
        fndm,
    },
    {
        "Two Sliding Window",
        "tsw",
        tsw,
    },
    {
        "Backward Fast Search",
        "bfs",
        bfs,
    },
    {
        "BNDM with loop unrolling",
        "bndmq2",
        bndmq2,
    },
    {
        "Simplified BNDM with Horspool Shift",
        "sbndm_bmh",
        sbndm_bmh,
    },
    {
        "Boyer Moore Horspool using Probabilites",
        "pbmh",
        pbmh,
    },
    {
        "Turbo Reverse Factor",
        "trf",
        trf,
    },
    {
        "Bitparallel Wide Window",
        "bww",
        bww,
    },
    {
        "Extended Backward Oracle Matching",
        "ebom",
        ebom,
    },
    {
        "Forward Backward Oracle Matching",
        "fbom",
        fbom,
    },
    {
        "Simplified Extended Backward Oracle Matching",
        "sebom",
        sebom,
    },
    {
     	"Simplified Forward Backward Oracle Matching",
    	"sfbom",
        sfbom,
    },
    {
        "Forward Fast Search",
        "ffs",
        ffs0,
    },
    {
        "Horspool with BNDM test",
        "bmh_sbndm",
        bmh_sbndm,
    },
    {
        "Fast Average Optimal Shift Or",
        "faoso2",
        faoso2,
    },
    {
        "Average Optimal Shift Or",
        "aoso2",
        aoso2,
    },
    {
        "Improved Linear DAWG Matching 1",
        "ildm1",
        ildm1,
    },
    {
        "Improved Linear DAWG Matching 2",
        "ildm2",
        ildm2,
    },
    {
        "Small Alphabet Bit Parallel",
        "sabp",
        sabp,
    },
    {
        "BNDM with Extended Shift",
        "bxs",
        bxs,
    },
    {
        "Backward SNR DAWG Matching",
        "bsdm",
        bsdm,
    },
    {
        "Factorized BNDM",
        "kbndm",
        kbndm,
    },
    {
        "Factorized Shift And",
        "ksa",
        ksa,
    },
    {
        "Wu Manber for Single Pattern Matching",
        "hash3",
        hash3,
    },
    {
        "Bit Parallel Length Invariant Matcher",
        "blim",
        blim,
    },
    {
        "Forward Simplified BNDM",
        "fsbndm",
        fsbndm,
    },
    {
        "Backward Nondet. DAWG Matching with q-grams",
        "bndmq4",
        bndmq4,
    },
    {
        "Simplified BNDM with q-grams",
        "sbndmq2",
        sbndmq2,
    },
    {
        "Forward SBNDM with q-grams & s-f characters",
        "fsbndmq20",
        fsbndmq20,
    },
};

int algorithm_list_len = sizeof(algorithm_list)/sizeof(algorithm_list[0]);

#ifdef __cplusplus
}
#endif
#endif
