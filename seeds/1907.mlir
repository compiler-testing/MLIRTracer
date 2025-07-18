module {
  func.func @main(%arg0: tensor<28x88x96x43x13x23xf32>, %arg1: tensor<1x88x96x1x13x1xf32>, %arg2: tensor<57x64x42xi16>, %arg3: tensor<58x56x70x63x28x39xi32>, %arg4: tensor<1x56x1x1x28x1xi32>) -> (tensor<28x88x96x43x13x23xf32>, tensor<57x1x42xi16>, tensor<58x56x70x63x28x39xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<28x88x96x43x13x23xf32>, tensor<1x88x96x1x13x1xf32>) -> tensor<28x88x96x43x13x23xf32>
    %1 = tosa.reduce_product %arg2 {axis = 1 : i32} : (tensor<57x64x42xi16>) -> tensor<57x1x42xi16>
    %2 = tosa.intdiv %arg3, %arg4 : (tensor<58x56x70x63x28x39xi32>, tensor<1x56x1x1x28x1xi32>) -> tensor<58x56x70x63x28x39xi32>
    %3 = tosa.equal %2, %2 : (tensor<58x56x70x63x28x39xi32>, tensor<58x56x70x63x28x39xi32>) -> tensor<58x56x70x63x28x39xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<58x56x70x63x28x39xi1>, tensor<58x56x70x63x28x39xi1>) -> tensor<58x56x70x63x28x39xi1>
    %5 = tosa.logical_not %4 : (tensor<58x56x70x63x28x39xi1>) -> tensor<58x56x70x63x28x39xi1>
    %6 = tosa.bitwise_and %3, %5 : (tensor<58x56x70x63x28x39xi1>, tensor<58x56x70x63x28x39xi1>) -> tensor<58x56x70x63x28x39xi1>
    return %0, %1, %6 : tensor<28x88x96x43x13x23xf32>, tensor<57x1x42xi16>, tensor<58x56x70x63x28x39xi1>
  }
}
