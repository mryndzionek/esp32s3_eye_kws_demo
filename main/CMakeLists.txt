if(CONFIG_RNN_CELL_FASTGRNN)
    set(FAST_RNN_C_FILE "fast_grnn.c")
elseif(CONFIG_RNN_CELL_FASTRNN)
    set(FAST_RNN_C_FILE "fast_rnn.c")
else()
    message(FATAL "RNN cell type not set!")
endif()

idf_component_register(SRCS "fbank.c" ${FAST_RNN_C_FILE}
                            "esp32s3_eye_kws_demo.c"
                    PRIV_REQUIRES esp_driver_i2s
                                  esp_driver_gpio
                                  esp_driver_ledc
                                  spi_flash
                                  esp_timer
                                  esp_lcd
                    INCLUDE_DIRS "")

