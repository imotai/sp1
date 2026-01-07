add_library(scan_objs OBJECT
    scan.cu
)
target_link_libraries(scan_objs PRIVATE csl_common)
