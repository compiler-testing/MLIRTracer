module {
  func.func @main(%arg0: tensor<13x97x19x11x14x46xf32>, %arg1: tensor<6x2xi64>, %arg2: tensor<77x12x94x98x82xi1>, %arg3: tensor<1x1x94x98x82xi1>, %arg4: tensor<75x55x29x70xi16>, %arg5: tensor<78xi32>, %arg6: tensor<1xi32>) -> (tensor<13x97x19x11x14x92xf32>, tensor<77x12x94x98x82xi1>, tensor<75x55x1x70xi16>, tensor<78xi32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<13x97x19x11x14x46xf32>, !tosa.shape<12>, tensor<1xf32>) -> tensor<13x97x19x11x14x46xf32>
    %1 = tosa.sigmoid %0 : (tensor<13x97x19x11x14x46xf32>) -> tensor<13x97x19x11x14x46xf32>
    %2 = tosa.concat %1, %0 {axis = 5 : i32} : (tensor<13x97x19x11x14x46xf32>, tensor<13x97x19x11x14x46xf32>) -> tensor<13x97x19x11x14x92xf32>
    %3 = tosa.logical_or %arg2, %arg3 : (tensor<77x12x94x98x82xi1>, tensor<1x1x94x98x82xi1>) -> tensor<77x12x94x98x82xi1>
    %4 = tosa.reduce_max %arg4 {axis = 2 : i32} : (tensor<75x55x29x70xi16>) -> tensor<75x55x1x70xi16>
    %5 = tosa.intdiv %arg5, %arg6 : (tensor<78xi32>, tensor<1xi32>) -> tensor<78xi32>
    return %2, %3, %4, %5 : tensor<13x97x19x11x14x92xf32>, tensor<77x12x94x98x82xi1>, tensor<75x55x1x70xi16>, tensor<78xi32>
  }
}
