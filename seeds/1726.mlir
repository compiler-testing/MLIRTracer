module {
  func.func @main(%arg0: tensor<66x38x40x27x34x15xf32>, %arg1: tensor<66x38x40x27x51x15xf32>) -> tensor<66x38x40x27x85x15xf32> {
    %0 = tosa.concat %arg0, %arg1 {axis = 4 : i32} : (tensor<66x38x40x27x34x15xf32>, tensor<66x38x40x27x51x15xf32>) -> tensor<66x38x40x27x85x15xf32>
    return %0 : tensor<66x38x40x27x85x15xf32>
  }
}
