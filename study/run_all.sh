#!/bin/bash

auto_mode=false
verbose=false

for i in "$@"; do
  case $i in
    --auto)
      echo 'auto mode'
      auto_mode=true
      shift # past argument=value
      ;;
    --debug)
      echo 'debug mode'
      verbose='debug'
      shift # past argument=value
      ;;
    --dumb)
      echo 'dumb mode'
      verbose='dumb'
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

for x in cf/*toml
do
    echo "Executing $x"
  if [ "$verbose" == "debug" ]; then
    postfix="--debug"

  elif [ "$verbose" == "dumb" ]; then
    postfix="--dumb"
  else
    postfix=""
  fi
  cmd="python -m adagdvb $x $postfix"
  echo $cmd
  $cmd
  if [ "$auto_mode" == false ]; then
    read -p "Press any key to continue... " -n1 -s
  fi
  echo
done
