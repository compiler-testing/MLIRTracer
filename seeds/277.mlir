module {
  func.func @main(%arg0: tensor<56x46x32x42xi1>, %arg1: tensor<83x14x96x16xf32>) -> (tensor<56x1x32x42xi1>, tensor<1x14x96x16xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<56x46x32x42xi1>) -> tensor<56x1x32x42xi1>
    %1 = tosa.sigmoid %arg1 : (tensor<83x14x96x16xf32>) -> tensor<83x14x96x16xf32>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<83x14x96x16xf32>) -> tensor<1x14x96x16xf32>
    return %0, %2 : tensor<56x1x32x42xi1>, tensor<1x14x96x16xf32>
  }
}
