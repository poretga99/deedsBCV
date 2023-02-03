add_executable(deedsBCV "main.cpp" "deedsBCV.cpp")
target_link_libraries(deedsBCV PRIVATE 
	${argparse_LIBRARIES}
	${TBB_LIBRARIES}
	nifti::nifti
)

add_executable(test "main2.cpp")
target_link_libraries(test PRIVATE 
	${argparse_LIBRARIES}
)