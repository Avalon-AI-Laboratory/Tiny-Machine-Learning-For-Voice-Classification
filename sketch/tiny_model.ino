#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_256_new.h"

tflite::MicroErrorReporter tflErrorReporter; // Error Reporter
tflite::ErrorReporter* error_reporter = &tflErrorReporter;

tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* model = nullptr; // For loading model
tflite::MicroInterpreter* tflInterpreter = nullptr; // Interpreter

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensorArenaSize = 1024*54;
byte tensorArena[tensorArenaSize];

void setup() {
  model = tflite::GetModel(optimized_age_model_256_128_new_tflite);
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
  float values_input[46] = {-2.3517101 ,  3.28049675,  1.70005558, -1.60870113,  1.06089132,
                            1.96845785, -1.63522228,  0.2276715 , -1.17771445, -0.84859355,
                            1.08632598, -1.34553314, -1.1500658 , -0.81816235, -0.37633929,
                            0.24799274,  1.96172823,  2.51620562, -0.7750686 , -0.34625655,
                            0.51380715,  0.94830725, -1.17034049,  0.01587338,  0.4392351 ,
                          -0.65428423,  0.44047243, -0.36434684,  0.55989584, -0.23495585,
                            0.88449875,  0.13561021,  0.82691867, -0.998904  , -0.42994037,
                            0.11755078,  0.52454984,  0.21223954, -0.33420518,  0.26332118,
                            0.0442574 ,  0.16969875, -0.41782506,  0.05671601, -0.01269465,
                          -0.27382602};

  for (int i = 0; i < 46; i++) {
    input->data.f[i] = values_input[i];
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed!");
    return;
  }

  for (int i = 0; i < 8; i++) {
    Serial.println(output->data.f[i]);
  }
  Serial.println("==================================");
}
