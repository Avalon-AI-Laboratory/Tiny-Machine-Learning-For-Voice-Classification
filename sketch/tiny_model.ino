#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_data.h"

tflite::MicroErrorReporter tflErrorReporter;
tflite::ErrorReporter* error_reporter = &tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensorArenaSize = 4 * 1024;
byte tensorArena[tensorArenaSize];

const char* AGES[] = {
  "ones",
  "twos",
  "threes",
  "fours",
  "fives",
  "sixs",
  "sevens",
  "eights"
};

#define NUM_CLASSES (sizeof(AGES) / sizeof(AGES[0]))


void setup() {
  Serial.begin(9600);
  while (!Serial) {
    Serial.println("Error, you entered incorrect serial");
    delay(1000);
  }

  const tflite::Model* model = ::tflite::GetModel(optimized_quantized_repdatagen_age_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  tflInterpreter = new tflite::MicroInterpreter(model, tflOpsResolver, tensorArena, tensorArenaSize, error_reporter);
  tflInterpreter->AllocateTensors();
  input = tflInterpreter->input(0);
  output = tflInterpreter->output(0);
}

void loop() {
  float values_input[46] = {0.2134, 0.1234, 0.5632, 0.6882, 0.2345,
                            0.2552, 0.5522, 0.5121, 0.5122, 0.3445,
                            0.5995, 0.0996, 0.4404, 0.3569, 0.0012,
                            0.9491, 0.2244, 0.1120, 0.4481, 0.3441,
                            0.2134, 0.1234, 0.5632, 0.6882, 0.2345,
                            0.2552, 0.5522, 0.5121, 0.5122, 0.3445,
                            0.5995, 0.0996, 0.4404, 0.3569, 0.0012,
                            0.9491, 0.2244, 0.1120, 0.4481, 0.3441,
                            0.5223, 0.3588, 0.1112, 0.9955, 0.0224,
                            0.1291};

  for (int i = 0; i < 46; i++) {
    input->data.int8[i] = values_input[i] / input->params.scale + input->params.zero_point;
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }
  
  int8_t y[8];

  for (int i = 0; i < 8; i++) {
    y[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    Serial.println(y[i]);
  }
}
