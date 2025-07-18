module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<61x57x21x34xi64>, %arg2: tensor<1x57x1x1xi64>, %arg3: tensor<17xf32>) -> (tensor<i16>, tensor<17xf32>, tensor<61x57x21x34xi1>) {
    %0 = tosa.clz %arg0 : (tensor<i16>) -> tensor<i16>
    %1 = tosa.identity %0 : (tensor<i16>) -> tensor<i16>
    %2 = tosa.greater %arg1, %arg2 : (tensor<61x57x21x34xi64>, tensor<1x57x1x1xi64>) -> tensor<61x57x21x34xi1>
    %3 = tosa.sigmoid %arg3 : (tensor<17xf32>) -> tensor<17xf32>
    %4 = tosa.bitwise_not %2 : (tensor<61x57x21x34xi1>) -> tensor<61x57x21x34xi1>
    return %1, %3, %4 : tensor<i16>, tensor<17xf32>, tensor<61x57x21x34xi1>
  }
}
