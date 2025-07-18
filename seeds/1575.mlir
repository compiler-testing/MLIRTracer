module {
  func.func @main(%arg0: tensor<21x38xi64>, %arg1: tensor<39x38xi64>, %arg2: tensor<97x79x34x95x73x79xf32>) -> (tensor<60x38xi64>, tensor<97x79x34x95x73x79xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<21x38xi64>, tensor<39x38xi64>) -> tensor<60x38xi64>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<60x38xi64>, tensor<60x38xi64>) -> tensor<60x38xi64>
    %2 = tosa.exp %arg2 : (tensor<97x79x34x95x73x79xf32>) -> tensor<97x79x34x95x73x79xf32>
    %3 = tosa.abs %2 : (tensor<97x79x34x95x73x79xf32>) -> tensor<97x79x34x95x73x79xf32>
    %4 = tosa.abs %1 : (tensor<60x38xi64>) -> tensor<60x38xi64>
    %5 = tosa.log %3 : (tensor<97x79x34x95x73x79xf32>) -> tensor<97x79x34x95x73x79xf32>
    return %4, %5 : tensor<60x38xi64>, tensor<97x79x34x95x73x79xf32>
  }
}
