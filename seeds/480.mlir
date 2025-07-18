module {
  func.func @main(%arg0: tensor<44x64xi1>, %arg1: tensor<67x84x68x30x30x60xf32>) -> (tensor<67x84x68x30x30x60xi1>, tensor<1x64xi1>, tensor<67x84x68x30x30x60xf32>, tensor<67x84x68x30x30x60xf32>, tensor<1xi32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<44x64xi1>) -> tensor<44x64xi1>
    %1 = tosa.floor %arg1 : (tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xf32>
    %2 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<44x64xi1>) -> tensor<1x64xi1>
    %3 = tosa.tanh %1 : (tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xf32>
    %4 = tosa.equal %1, %3 : (tensor<67x84x68x30x30x60xf32>, tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xi1>
    %5 = tosa.floor %3 : (tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xf32>
    %6 = tosa.log %5 : (tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xf32>
    %7 = tosa.reduce_any %2 {axis = 1 : i32} : (tensor<1x64xi1>) -> tensor<1x1xi1>
    %8 = tosa.bitwise_and %4, %4 : (tensor<67x84x68x30x30x60xi1>, tensor<67x84x68x30x30x60xi1>) -> tensor<67x84x68x30x30x60xi1>
    %9 = tosa.argmax %7 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1xi32>
    %10 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<44x64xi1>) -> tensor<1x64xi1>
    %11 = tosa.intdiv %9, %9 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %12 = tosa.floor %1 : (tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xf32>
    %13 = tosa.pow %6, %1 : (tensor<67x84x68x30x30x60xf32>, tensor<67x84x68x30x30x60xf32>) -> tensor<67x84x68x30x30x60xf32>
    %14 = tosa.reduce_product %11 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    return %8, %10, %12, %13, %14 : tensor<67x84x68x30x30x60xi1>, tensor<1x64xi1>, tensor<67x84x68x30x30x60xf32>, tensor<67x84x68x30x30x60xf32>, tensor<1xi32>
  }
}
