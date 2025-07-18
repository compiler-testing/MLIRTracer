module {
  func.func @main(%arg0: tensor<28xi16>) -> tensor<1xi16> {
    %s_0_start = tosa.const_shape {values = dense<[ 24 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<28xi16>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xi16>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<3xi16>) -> tensor<1xi16>
    %2 = tosa.add %1, %1 : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi16>
    return %2 : tensor<1xi16>
  }
}
