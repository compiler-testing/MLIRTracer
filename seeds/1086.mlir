module {
  func.func @main(%arg0: tensor<64x91xi16>, %arg1: tensor<17x62x34xi8>, %arg2: tensor<17x1x1xi8>, %arg3: tensor<77xi32>, %arg4: tensor<77xi32>, %arg5: tensor<75x43x86xf32>) -> (tensor<1x91xi16>, tensor<77xi32>, tensor<75x43x86xf32>, tensor<17x62x34xi1>, tensor<75x43x86xf32>) {
    %0 = tosa.abs %arg0 : (tensor<64x91xi16>) -> tensor<64x91xi16>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<64x91xi16>) -> tensor<1x91xi16>
    %2 = tosa.bitwise_and %1, %1 : (tensor<1x91xi16>, tensor<1x91xi16>) -> tensor<1x91xi16>
    %3 = tosa.greater %arg1, %arg2 : (tensor<17x62x34xi8>, tensor<17x1x1xi8>) -> tensor<17x62x34xi1>
    %4 = tosa.maximum %arg3, %arg4 : (tensor<77xi32>, tensor<77xi32>) -> tensor<77xi32>
    %5 = tosa.reciprocal %arg5 : (tensor<75x43x86xf32>) -> tensor<75x43x86xf32>
    %6 = tosa.rsqrt %5 : (tensor<75x43x86xf32>) -> tensor<75x43x86xf32>
    %7 = tosa.clz %3 : (tensor<17x62x34xi1>) -> tensor<17x62x34xi1>
    %8 = tosa.ceil %5 : (tensor<75x43x86xf32>) -> tensor<75x43x86xf32>
    return %2, %4, %6, %7, %8 : tensor<1x91xi16>, tensor<77xi32>, tensor<75x43x86xf32>, tensor<17x62x34xi1>, tensor<75x43x86xf32>
  }
}
