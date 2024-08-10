#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "sdkconfig.h"

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_system.h"
#include "esp_log.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "driver/ledc.h"

#include "esp_check.h"
#include "esp_timer.h"
#include "esp_task.h"

#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"

#include "fbank.h"
#include "fast_rnn.h"
#include "bmp.h"

#define MIC_STD_BCLK_IO1 (GPIO_NUM_41)
#define MIC_STD_WS_IO1 (GPIO_NUM_42)
#define MIC_STD_DOUT_IO1 (GPIO_NUM_2)

#define BOARD_LCD_MOSI (GPIO_NUM_47)
#define BOARD_LCD_MISO (GPIO_NUM_NC)
#define BOARD_LCD_SCK (GPIO_NUM_21)
#define BOARD_LCD_CS (GPIO_NUM_44)
#define BOARD_LCD_DC (GPIO_NUM_43)
#define BOARD_LCD_RST (GPIO_NUM_NC)
#define BOARD_LCD_BACKLIGHT (GPIO_NUM_48)

#define BOARD_LCD_H_RES (240)
#define BOARD_LCD_V_RES (240)
#define BOARD_LCD_CMD_BITS (8)
#define BOARD_LCD_PARAM_BITS (8)

#define BOARD_LCD_PIXEL_CLOCK_HZ (60 * 1000 * 1000)

#define MIC_CHUNK_SIZE (SAMPLE_LEN / 4)
#define MIC_OFFSET (SAMPLE_LEN - MIC_CHUNK_SIZE)

#define DETECTION_THRESHOLD (3.5f)

typedef struct
{
    QueueHandle_t mic_q;
    SemaphoreHandle_t sync_sema;
} mic_task_ctx_t;

static const char *TAG = "esp32s3_eye_kws_demo";

// clang-format off
static const uint16_t palette[256] = {
    0x0000, 0x0000, 0x0100, 0x2100, 0x2200, 0x2200, 0x2300, 0x4308, 0x4408, 0x4408, 0x6508, 0x6508, 0x6608, 0x8610, 0x8710, 0x8710,
    0x8810, 0xa810, 0xa910, 0xa910, 0xca18, 0xca18, 0xcb18, 0xeb18, 0xec18, 0xec18, 0xed18, 0x0d21, 0x0e21, 0x0e21, 0x2f21, 0x2f21,
    0x3021, 0x3029, 0x3029, 0x3029, 0x3129, 0x3129, 0x3129, 0x3129, 0x3231, 0x3231, 0x3231, 0x3231, 0x3331, 0x3331, 0x3331, 0x3339,
    0x3439, 0x3439, 0x3439, 0x3439, 0x3539, 0x3541, 0x3541, 0x3541, 0x3641, 0x3641, 0x3641, 0x3641, 0x3749, 0x3749, 0x3749, 0x3749,
    0x3749, 0x3751, 0x3751, 0x5751, 0x5651, 0x5659, 0x5659, 0x5659, 0x5561, 0x5561, 0x5561, 0x5569, 0x5469, 0x7469, 0x7469, 0x7471,
    0x7371, 0x7371, 0x7379, 0x7379, 0x7279, 0x7281, 0x7281, 0x9281, 0x9181, 0x9189, 0x9189, 0x9189, 0x9091, 0x9091, 0x9091, 0x9091,
    0x8f99, 0xaf99, 0xafa1, 0xaea1, 0xaea1, 0xaea9, 0xada9, 0xadb1, 0xadb1, 0xacb1, 0xacb9, 0xccb9, 0xcbc1, 0xcbc1, 0xcac1, 0xcac9,
    0xcac9, 0xc9d1, 0xc9d1, 0xc9d1, 0xc8d9, 0xe8d9, 0xe8e1, 0xe7e1, 0xe7e1, 0xe7e9, 0xe6e9, 0xe6f1, 0xe6f1, 0xe5f1, 0xe5f9, 0x05fa,
    0x04fa, 0x04fa, 0x24fa, 0x24fa, 0x44fa, 0x44fa, 0x63fa, 0x63fa, 0x83fa, 0x83f2, 0xa3f2, 0xa3f2, 0xc2f2, 0xc2f2, 0xe2f2, 0xe2f2,
    0x02f3, 0x02f3, 0x22f3, 0x21eb, 0x41eb, 0x41eb, 0x61eb, 0x61eb, 0x81eb, 0x81eb, 0xa0eb, 0xa0eb, 0xc0eb, 0xc0e3, 0xe0e3, 0xe0e3,
    0x00e4, 0x00e4, 0x20e4, 0x20e4, 0x40e4, 0x40e4, 0x60e4, 0x60e4, 0x80e4, 0x81e4, 0xa1e4, 0xa1e4, 0xc1e4, 0xc1e4, 0xe1e4, 0xe1e4,
    0x01e5, 0x01e5, 0x21e5, 0x22e5, 0x42e5, 0x42e5, 0x62e5, 0x62e5, 0x82e5, 0x82e5, 0xa2e5, 0xa2e5, 0xc2e5, 0xc3e5, 0xe3e5, 0xe3e5,
    0x03e6, 0x03e6, 0x04e6, 0x24e6, 0x25e6, 0x25e6, 0x25e6, 0x46e6, 0x46e6, 0x47e6, 0x67e6, 0x67e6, 0x68e6, 0x88e6, 0x89e6, 0x89e6,
    0x89e6, 0xaae6, 0xaae6, 0xabe6, 0xcbe6, 0xcbe6, 0xcce6, 0xece6, 0xede6, 0xede6, 0xede6, 0x0ee7, 0x0ee7, 0x0fe7, 0x2fe7, 0x2fe7,
    0x30e7, 0x30e7, 0x51ef, 0x51ef, 0x52ef, 0x52ef, 0x53ef, 0x73ef, 0x74ef, 0x74ef, 0x75ef, 0x75ef, 0x96f7, 0x96f7, 0x97f7, 0x97f7,
    0x98f7, 0xb8f7, 0xb9f7, 0xb9f7, 0xbaf7, 0xbaf7, 0xdbff, 0xdbff, 0xdcff, 0xdcff, 0xddff, 0xfdff, 0xfeff, 0xfeff, 0xffff, 0xffff,
};
// clang-format on

static i2s_chan_handle_t rx_chan;
static int32_t r_buf[2][MIC_CHUNK_SIZE];
static float input[SAMPLE_LEN];
static float features[NUM_FRAMES][NUM_FILT];

static esp_lcd_panel_handle_t panel_handle;
static uint16_t *line_buf;

static void init_mic(void)
{
    i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &rx_chan));

    i2s_std_config_t rx_std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED, // some codecs may require mclk signal, this example doesn't need it
            .bclk = MIC_STD_BCLK_IO1,
            .ws = MIC_STD_WS_IO1,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_STD_DOUT_IO1,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    rx_std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &rx_std_cfg));
}

static void draw_bmp(const uint32_t *bmp, bool full)
{
    const int ry = full ? BOARD_LCD_V_RES : BOARD_LCD_V_RES - (2 * NUM_FILT);
    const int rx = full ? BOARD_LCD_H_RES : BOARD_LCD_H_RES - (2 * NUM_FILT);
    const int ox = (BOARD_LCD_H_RES - rx) / 2;

    for (int y = 0; y < ry; y++)
    {
        for (size_t x = 0; x < rx; x++)
        {
            bool bit = bmp[(y * ((rx + 32 - 1) / 32)) + (x / 32)] & (1UL << (31 - (x % 32)));
            line_buf[x] = bit ? 0xFFFF : 0x0000;
        }
        esp_lcd_panel_draw_bitmap(panel_handle, ox, y, ox + rx, y + 1, line_buf);
    }
}

static void draw_features(float features[NUM_FRAMES][NUM_FILT])
{
    for (size_t i = 0; i < NUM_FILT; i++)
    {
        for (size_t j = 0; j < BOARD_LCD_H_RES; j++)
        {
            int32_t idx = ((features[(j * NUM_FRAMES) / BOARD_LCD_H_RES][i] + 0.3) * 255) / 2.0;
            if (idx < 0)
            {
                idx = 0;
            }
            if (idx > 255)
            {
                idx = 255;
            }
            line_buf[j] = palette[idx];
        }
        esp_lcd_panel_draw_bitmap(panel_handle, 0, BOARD_LCD_V_RES - 2 * i, BOARD_LCD_H_RES, BOARD_LCD_V_RES - (2 * i) + 1, line_buf);
        esp_lcd_panel_draw_bitmap(panel_handle, 0, BOARD_LCD_V_RES - (2 * i) + 1, BOARD_LCD_H_RES, BOARD_LCD_V_RES - (2 * i) + 2, line_buf);
    }
}

static void clear_disp(void)
{
    for (size_t x = 0; x < BOARD_LCD_H_RES; x++)
    {
        line_buf[x] = 0x0000;
    }

    for (int y = 0; y < BOARD_LCD_V_RES; y++)
    {
        esp_lcd_panel_draw_bitmap(panel_handle, 0, y, BOARD_LCD_H_RES, y + 1, line_buf);
    }
}

esp_err_t bsp_display_brightness_init(void)
{
    // Setup LEDC peripheral for PWM backlight control
    const ledc_channel_config_t LCD_backlight_channel = {
        .gpio_num = BOARD_LCD_BACKLIGHT,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_0,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = 1,
        .duty = 0,
        .hpoint = 0,
        .flags.output_invert = true};
    const ledc_timer_config_t LCD_backlight_timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_10_BIT,
        .timer_num = 1,
        .freq_hz = 5000,
        .clk_cfg = LEDC_AUTO_CLK};

    ESP_ERROR_CHECK(ledc_timer_config(&LCD_backlight_timer));
    ESP_ERROR_CHECK(ledc_channel_config(&LCD_backlight_channel));

    return ESP_OK;
}

esp_err_t bsp_display_brightness_set(int brightness_percent)
{
    if (brightness_percent > 100)
    {
        brightness_percent = 100;
    }
    else if (brightness_percent < 0)
    {
        brightness_percent = 0;
    }

    ESP_LOGI(TAG, "Setting LCD backlight: %d%%", brightness_percent);
    // LEDC resolution set to 10bits, thus: 100% = 1023
    uint32_t duty_cycle = (1023 * brightness_percent) / 100;
    ESP_ERROR_CHECK(ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, duty_cycle));
    ESP_ERROR_CHECK(ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0));

    return ESP_OK;
}

static void init_lcd(void)
{
    ESP_LOGI(TAG, "Initialize SPI bus");
    spi_bus_config_t bus_conf = {
        .mosi_io_num = BOARD_LCD_MOSI,
        .miso_io_num = BOARD_LCD_MISO,
        .sclk_io_num = BOARD_LCD_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_conf, SPI_DMA_CH_AUTO));

    ESP_LOGD(TAG, "Install panel IO");
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_spi_config_t io_config = {
        .cs_gpio_num = BOARD_LCD_CS,
        .dc_gpio_num = BOARD_LCD_DC,
        .spi_mode = 0,
        .pclk_hz = BOARD_LCD_PIXEL_CLOCK_HZ,
        .trans_queue_depth = 10,
        .lcd_cmd_bits = BOARD_LCD_CMD_BITS,
        .lcd_param_bits = BOARD_LCD_PARAM_BITS,
    };
    // Attach the LCD to the SPI bus
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI2_HOST, &io_config, &io_handle));

    ESP_LOGD(TAG, "Install ST7789 panel driver");
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = BOARD_LCD_RST,
        .rgb_endian = LCD_RGB_ENDIAN_RGB,
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    esp_lcd_panel_invert_color(panel_handle, true); // Set inversion for esp32s3eye

    // turn on display
    esp_lcd_panel_disp_on_off(panel_handle, true);
    bsp_display_brightness_init();
    bsp_display_brightness_set(100);
}

static void mic_stream_task(void *args)
{
    size_t hand = 0;
    size_t r_bytes = 0;
    mic_task_ctx_t *mic_task_ctx = args;

    init_mic();
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));

    ESP_LOGI(TAG, "Mic task started");

    while (true)
    {
        if (i2s_channel_read(rx_chan, r_buf[hand], MIC_CHUNK_SIZE * sizeof(int32_t), &r_bytes, 500) == ESP_OK)
        {
            assert(r_bytes == (MIC_CHUNK_SIZE * sizeof(int32_t)));

            // 500ms in the above call should be more than enough
            // to receive full chunk. Not receiving it most likely
            // signifies a bug.
            if (xSemaphoreTake(mic_task_ctx->sync_sema, 0) == pdFALSE)
            {
                ESP_LOGW(TAG, "Real-time constraint between the mic task and main task violated.");
            }

            int32_t *item = r_buf[hand];
            BaseType_t res = xQueueSend(mic_task_ctx->mic_q, &item, 0);
            assert(res == pdPASS);
            hand ^= 1;
        }
        else
        {
            ESP_LOGE(TAG, "Read Task: i2s read failed");
        }
    }
}

void app_main(void)
{
    size_t label;
    float logit;
    int64_t ts;
    int32_t *data;
    size_t debounce_count = 0;
    uint32_t count = 0;

    /* Print chip information */
    esp_chip_info_t chip_info;
    uint32_t flash_size;
    esp_chip_info(&chip_info);
    ESP_LOGI(TAG, "This is %s chip with %d CPU core(s), %s%s%s%s, ",
             CONFIG_IDF_TARGET,
             chip_info.cores,
             (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "WiFi/" : "",
             (chip_info.features & CHIP_FEATURE_BT) ? "BT" : "",
             (chip_info.features & CHIP_FEATURE_BLE) ? "BLE" : "",
             (chip_info.features & CHIP_FEATURE_IEEE802154) ? ", 802.15.4 (Zigbee/Thread)" : "");

    unsigned major_rev = chip_info.revision / 100;
    unsigned minor_rev = chip_info.revision % 100;
    ESP_LOGI(TAG, "silicon revision v%d.%d, ", major_rev, minor_rev);
    if (esp_flash_get_size(NULL, &flash_size) != ESP_OK)
    {
        ESP_LOGE(TAG, "Get flash size failed");
        return;
    }

    ESP_LOGI(TAG, "%" PRIu32 "MB %s flash", flash_size / (uint32_t)(1024 * 1024),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");

    ESP_LOGI(TAG, "Minimum free heap size: %" PRIu32 " bytes", esp_get_minimum_free_heap_size());

    line_buf = (uint16_t *)heap_caps_malloc((BOARD_LCD_H_RES) * sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    assert(line_buf);
    init_lcd();
    draw_bmp((const uint32_t *)title_bmp, true);
    vTaskDelay(pdMS_TO_TICKS(5000));
    clear_disp();
    draw_bmp((const uint32_t *)mic_bmp, false);

    mic_task_ctx_t mic_task_ctx;

    mic_task_ctx.mic_q = xQueueCreate(1, sizeof(int32_t *));
    assert(mic_task_ctx.mic_q);
    mic_task_ctx.sync_sema = xSemaphoreCreateBinary();
    assert(mic_task_ctx.sync_sema);

    xTaskCreatePinnedToCore(mic_stream_task, "mic_stream_task", 2 * 4096, &mic_task_ctx,
                            ESP_TASK_MAIN_PRIO + 1, NULL, CONFIG_FREERTOS_NUMBER_OF_CORES - 1);

    while (true)
    {
        xSemaphoreGive(mic_task_ctx.sync_sema);
        if (xQueueReceive(mic_task_ctx.mic_q, &data, portMAX_DELAY) == pdPASS)
        {
            ESP_LOGD(TAG, "Received mic data");
            ts = esp_timer_get_time();

            memmove(input, &input[MIC_CHUNK_SIZE], MIC_OFFSET * sizeof(float));

            for (size_t i = 0; i < MIC_CHUNK_SIZE; i++)
            {
                input[MIC_OFFSET + i] = (float)(data[i] >> 14);
            }

            fbank_prep(&input[MIC_OFFSET], MIC_CHUNK_SIZE);
            fbank(input, features);
            for (size_t i = 0; i < NUM_FRAMES; i++)
            {
                fbank_norm(features[i]);
            }

            nn_process(features, &logit, &label);

            ts = esp_timer_get_time() - ts;
            ESP_LOGD(TAG, "Infer took %lld ms", ts / 1000);
            if (debounce_count)
            {
                if (label == 0)
                {
                    debounce_count--;
                }
            }
            else
            {
                if (label > 0)
                {
                    ESP_LOGI(TAG, "label: '%s', label_idx: %u, logit: %f, inf_time: %lldms",
                             fbank_label_idx_to_str(label), label, logit, ts / 1000);
                    if (logit >= DETECTION_THRESHOLD)
                    {
                        ESP_LOGI(TAG,"Detection above threshold");
                        debounce_count = 3;
                        switch (label)
                        {
                        case 1:
                            draw_bmp((const uint32_t *)left_bmp, false);
                            break;

                        case 2:
                            draw_bmp((const uint32_t *)right_bmp, false);
                            break;

                        case 3:
                            draw_bmp((const uint32_t *)up_bmp, false);
                            break;

                        case 4:
                            draw_bmp((const uint32_t *)down_bmp, false);
                            break;

                        case 5:
                            draw_bmp((const uint32_t *)stop_bmp, false);
                            break;

                        default:
                            break;
                        }
                    }
                }
                else
                {
                    if (count % 4 < 2)
                    {
                        draw_bmp((const uint32_t *)mic_bmp, false);
                    }
                    else
                    {
                        draw_bmp((const uint32_t *)mic2_bmp, false);
                    }
                }
            }
            draw_features(features);
            count++;
        }
        else
        {
            assert(true);
        }
    }
}
