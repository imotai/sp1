#pragma once

extern "C" void* constraint_poly_eval_32_koala_bear_kernel();
extern "C" void* constraint_poly_eval_64_koala_bear_kernel();
extern "C" void* constraint_poly_eval_128_koala_bear_kernel();
extern "C" void* constraint_poly_eval_256_koala_bear_kernel();
extern "C" void* constraint_poly_eval_512_koala_bear_kernel();
extern "C" void* constraint_poly_eval_1024_koala_bear_kernel();

extern "C" void* constraint_poly_eval_32_koala_bear_extension_kernel();
extern "C" void* constraint_poly_eval_64_koala_bear_extension_kernel();
extern "C" void* constraint_poly_eval_128_koala_bear_extension_kernel();
extern "C" void* constraint_poly_eval_256_koala_bear_extension_kernel();
extern "C" void* constraint_poly_eval_512_koala_bear_extension_kernel();
extern "C" void* constraint_poly_eval_1024_koala_bear_extension_kernel();