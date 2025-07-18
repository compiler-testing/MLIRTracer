module {
  func.func @main(%arg0: tensor<41x30x59x92xi16>, %arg1: tensor<4x2xi64>, %arg2: tensor<74x47x82x75x11xi32>, %arg3: tensor<74x1x82x75x11xi32>) -> (tensor<41x30x59x92xi16>, tensor<74x47x82x75x11xi32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<41x30x59x92xi16>, !tosa.shape<8>, tensor<1xi16>) -> tensor<41x30x59x92xi16>
    %1 = tosa.bitwise_and %0, %0 : (tensor<41x30x59x92xi16>, tensor<41x30x59x92xi16>) -> tensor<41x30x59x92xi16>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<74x47x82x75x11xi32>, tensor<74x1x82x75x11xi32>) -> tensor<74x47x82x75x11xi32>
    %3 = tosa.add %2, %2 : (tensor<74x47x82x75x11xi32>, tensor<74x47x82x75x11xi32>) -> tensor<74x47x82x75x11xi32>
    %4 = tosa.clamp %3 {min_val = -46 : i32, max_val = -32 : i32} : (tensor<74x47x82x75x11xi32>) -> tensor<74x47x82x75x11xi32>
    return %1, %4 : tensor<41x30x59x92xi16>, tensor<74x47x82x75x11xi32>
  }
}
