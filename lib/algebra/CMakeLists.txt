add_library(algebra_objs OBJECT
    algebra.cu
)
target_link_libraries(algebra_objs PRIVATE csl_common)
