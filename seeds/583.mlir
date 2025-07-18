module {
  func.func @main(%arg0: tensor<80x31x47xf32>, %arg1: tensor<3x2xi32>) -> tensor<160x31x47xf32> {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<80x31x47xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<80x31x47xf32>
    %1 = tosa.log %0 : (tensor<80x31x47xf32>) -> tensor<80x31x47xf32>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<80x31x47xf32>, tensor<80x31x47xf32>) -> tensor<160x31x47xf32>
    return %2 : tensor<160x31x47xf32>
  }
}
