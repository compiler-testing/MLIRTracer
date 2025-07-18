module {
  func.func @main(%arg0: tensor<72x80x75x84x24x30xf32>, %arg1: tensor<25x77xi8>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<72x80x75x84x24x30xi1>, tensor<72x80x75x84x24x30xi1>, tensor<72x80x75x84x24x30xf32>, tensor<1x77xi1>, tensor<72x80x75x84x24x30xf32>, tensor<1x77xi8>, tensor<i32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<72x80x75x84x24x30xf32>) -> tensor<72x80x75x84x24x30xf32>
    %1 = tosa.equal %0, %0 : (tensor<72x80x75x84x24x30xf32>, tensor<72x80x75x84x24x30xf32>) -> tensor<72x80x75x84x24x30xi1>
    %2 = tosa.greater %0, %0 : (tensor<72x80x75x84x24x30xf32>, tensor<72x80x75x84x24x30xf32>) -> tensor<72x80x75x84x24x30xi1>
    %3 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<25x77xi8>) -> tensor<1x77xi8>
    %4 = tosa.clz %1 : (tensor<72x80x75x84x24x30xi1>) -> tensor<72x80x75x84x24x30xi1>
    %5 = tosa.greater %3, %3 : (tensor<1x77xi8>, tensor<1x77xi8>) -> tensor<1x77xi1>
    %6 = tosa.exp %0 : (tensor<72x80x75x84x24x30xf32>) -> tensor<72x80x75x84x24x30xf32>
    %7 = tosa.tanh %6 : (tensor<72x80x75x84x24x30xf32>) -> tensor<72x80x75x84x24x30xf32>
    %8 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<1x77xi1>) -> tensor<1x77xi1>
    %9 = tosa.sigmoid %0 : (tensor<72x80x75x84x24x30xf32>) -> tensor<72x80x75x84x24x30xf32>
    %10 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<1x77xi8>) -> tensor<1x77xi8>
    %11 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = tosa.add %11, %11 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %2, %4, %7, %8, %9, %10, %12 : tensor<72x80x75x84x24x30xi1>, tensor<72x80x75x84x24x30xi1>, tensor<72x80x75x84x24x30xf32>, tensor<1x77xi1>, tensor<72x80x75x84x24x30xf32>, tensor<1x77xi8>, tensor<i32>
  }
}
