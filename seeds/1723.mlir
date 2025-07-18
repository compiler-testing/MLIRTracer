module {
  func.func @main(%arg0: tensor<98x45x97x60xi8>, %arg1: tensor<37x2x61x51x42x85xi1>, %arg2: tensor<1x1x1x1x42x85xi1>) -> (tensor<45x97x60xi32>, tensor<37x2x61x51x42x85xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<98x45x97x60xi8>) -> tensor<45x97x60xi32>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<45x97x60xi32>, tensor<45x97x60xi32>) -> tensor<45x97x60xi32>
    %2 = tosa.logical_and %arg1, %arg2 : (tensor<37x2x61x51x42x85xi1>, tensor<1x1x1x1x42x85xi1>) -> tensor<37x2x61x51x42x85xi1>
    return %1, %2 : tensor<45x97x60xi32>, tensor<37x2x61x51x42x85xi1>
  }
}
