module {
  func.func @main(%arg0: tensor<52x81xi8>, %arg1: tensor<17x20x7x1x76x62xi32>, %arg2: tensor<1x20x7x1x1x1xi32>, %arg3: tensor<39xf32>, %arg4: tensor<39xf32>) -> (tensor<17x20x7x1x76x62xi32>, tensor<81xi32>, tensor<39xf32>) {
    %0 = tosa.clz %arg0 : (tensor<52x81xi8>) -> tensor<52x81xi8>
    %1 = tosa.greater_equal %0, %0 : (tensor<52x81xi8>, tensor<52x81xi8>) -> tensor<52x81xi1>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<17x20x7x1x76x62xi32>, tensor<1x20x7x1x1x1xi32>) -> tensor<17x20x7x1x76x62xi32>
    %3 = tosa.argmax %1 {axis = 0 : i32} : (tensor<52x81xi1>) -> tensor<81xi32>
    %4 = tosa.pow %arg3, %arg4 : (tensor<39xf32>, tensor<39xf32>) -> tensor<39xf32>
    return %2, %3, %4 : tensor<17x20x7x1x76x62xi32>, tensor<81xi32>, tensor<39xf32>
  }
}
