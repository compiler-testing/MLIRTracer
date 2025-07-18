module {
  func.func @main(%arg0: tensor<75x31x15x24xi16>, %arg1: tensor<38x81x91x2x69xf32>) -> (tensor<75x31x15x1xi16>, tensor<38653524xi1>, tensor<38x81x91x2x69xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<75x31x15x24xi16>) -> tensor<75x31x15x1xi16>
    %1 = tosa.exp %arg1 : (tensor<38x81x91x2x69xf32>) -> tensor<38x81x91x2x69xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<38x81x91x2x69xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<38x81x91x2x69xf32>
    %3 = tosa.sigmoid %2 : (tensor<38x81x91x2x69xf32>) -> tensor<38x81x91x2x69xf32>
    %r_4 = tosa.const_shape {values = dense<[ 38653524 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %3, %r_4 : (tensor<38x81x91x2x69xf32>, !tosa.shape<1>) -> tensor<38653524xf32>
    %5 = tosa.equal %4, %4 : (tensor<38653524xf32>, tensor<38653524xf32>) -> tensor<38653524xi1>
    %6 = tosa.rsqrt %2 : (tensor<38x81x91x2x69xf32>) -> tensor<38x81x91x2x69xf32>
    %7 = tosa.sub %6, %2 : (tensor<38x81x91x2x69xf32>, tensor<38x81x91x2x69xf32>) -> tensor<38x81x91x2x69xf32>
    %8 = tosa.minimum %7, %2 : (tensor<38x81x91x2x69xf32>, tensor<38x81x91x2x69xf32>) -> tensor<38x81x91x2x69xf32>
    return %0, %5, %8 : tensor<75x31x15x1xi16>, tensor<38653524xi1>, tensor<38x81x91x2x69xf32>
  }
}
