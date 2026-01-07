add_library(jagged_assist_objs OBJECT
    branching_program.cu
)
target_link_libraries(jagged_assist_objs PRIVATE csl_common)
