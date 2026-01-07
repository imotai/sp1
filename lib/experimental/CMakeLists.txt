add_library(experimental_objs OBJECT
    look_ahead.cu
)
target_link_libraries(experimental_objs PRIVATE csl_common)
