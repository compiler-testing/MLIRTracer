module {
  func.func @main(%arg0: tensor<45x90x55xi8>, %arg1: tensor<3x2xi64>, %arg2: tensor<30x83x22x53xf32>) -> (tensor<45x55xi32>, tensor<30x83x22x53xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<45x90x55xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<45x90x55xi8>
    %1 = tosa.minimum %0, %0 : (tensor<45x90x55xi8>, tensor<45x90x55xi8>) -> tensor<45x90x55xi8>
    %2 = tosa.argmax %1 {axis = 1 : i32} : (tensor<45x90x55xi8>) -> tensor<45x55xi32>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<45x55xi32>) -> tensor<45x55xi32>
    %4 = tosa.ceil %arg2 : (tensor<30x83x22x53xf32>) -> tensor<30x83x22x53xf32>
    return %3, %4 : tensor<45x55xi32>, tensor<30x83x22x53xf32>
  }
}
