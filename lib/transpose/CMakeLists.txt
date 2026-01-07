add_library(transpose_objs OBJECT
    transpose.cu
)
target_link_libraries(transpose_objs PRIVATE csl_common)
