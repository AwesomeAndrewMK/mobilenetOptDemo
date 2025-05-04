import React from 'react';
import {Text, SafeAreaView, StyleSheet, TouchableOpacity, Alert} from 'react-native';
import {loadTensorflowModel, TensorflowModel} from 'react-native-fast-tflite';

export const App = () => {
  const mobilenetv2_fp32 = loadTensorflowModel(
    require('./models/mobilenetv2_fp32.tflite'),
  );
  const mobilenetv2_fp16 = loadTensorflowModel(
    require('./models/mobilenetv2_fp16.tflite'),
  );
  const mobilenetv2_int8 = loadTensorflowModel(
    require('./models/mobilenetv2_int8.tflite'),
  );

  const input = new Float32Array(224 * 224 * 3).fill(0.5);
  const inputINT8 = new Uint8Array(224 * 224 * 3).fill(0.5);

  const loadModel = async (
    mobilenetv2Model: Promise<TensorflowModel>,
    inputItem: Float32Array | Uint8Array,
    label: string,
  ) => {
    try {
      const model = await mobilenetv2Model;
      const start = Date.now();
      await model.run([inputItem]);
      const end = Date.now();
      console.log(`${label} model inference time: ${end - start} ms`);
      Alert.alert(`${label} model inference time: ${end - start} ms`);
    } catch (error) {
      console.error('Error loading model:', error);
    }
  };

  return (
    <SafeAreaView style={styles.center}>
      <TouchableOpacity>
        <Text
          style={styles.button}
          onPress={async () => {
            await loadModel(mobilenetv2_fp32, input, 'FP32');
          }}>
          Load mobilenetv2_fp32
        </Text>
        <Text
          style={styles.button}
          onPress={async () => {
            await loadModel(mobilenetv2_fp16, input, 'FP16');
          }}>
          Load mobilenetv2_fp16
        </Text>
        <Text
          onPress={async () => {
            await loadModel(mobilenetv2_int8, inputINT8, 'INT8');
          }}>
          Load mobilenetv2_int8
        </Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  button: {
    paddingBottom: 16,
  },
});
