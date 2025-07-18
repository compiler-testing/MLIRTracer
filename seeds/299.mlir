module {
  func.func @main(%arg0: tensor<24x55x17xf32>) -> (tensor<1x55x17xi1>, tensor<24x55x17xf32>) {
    %0 = tosa.floor %arg0 : (tensor<24x55x17xf32>) -> tensor<24x55x17xf32>
    %1 = tosa.equal %0, %0 : (tensor<24x55x17xf32>, tensor<24x55x17xf32>) -> tensor<24x55x17xi1>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<24x55x17xi1>) -> tensor<1x55x17xi1>
    %3 = tosa.sigmoid %0 : (tensor<24x55x17xf32>) -> tensor<24x55x17xf32>
    return %2, %3 : tensor<1x55x17xi1>, tensor<24x55x17xf32>
  }
}
