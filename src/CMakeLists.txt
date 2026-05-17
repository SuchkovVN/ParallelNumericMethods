add_subdirectory(common)

foreach(proj IN LISTS BUILD_PROJECT)
        if(proj MATCHES "all")
            add_subdirectory("lab_*")
            break()
        endif()
        add_subdirectory(${proj})
endforeach()
