#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <model_data.h>

tflite::MicroErrorReporter tflErrorReporter;
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
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
  }

  tflModel = tflite::GetModel(optimized_quantized_repdatagen_age_model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  tflInterpreter->AllocateTensors();
  tflInputTensors = tflInterpreter->input(0);
  tflInputTensors = tflInterpreter->output(0);
}

void loop() {
  float input1, input2, input3, input4;

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }
  
  // Still on progress...
}
