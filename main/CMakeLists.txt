idf_component_register(SRCS "fbank.c" "fast_rnn.c" "esp32s3_eye_kws_demo.c"
                    PRIV_REQUIRES esp_driver_i2s
                                  esp_driver_gpio
                                  esp_driver_ledc
                                  spi_flash
                                  esp_timer
                                  esp_lcd
                    INCLUDE_DIRS "")
