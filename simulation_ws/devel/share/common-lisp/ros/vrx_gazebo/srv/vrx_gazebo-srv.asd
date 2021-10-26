
(cl:in-package :asdf)

(defsystem "vrx_gazebo-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ColorSequence" :depends-on ("_package_ColorSequence"))
    (:file "_package_ColorSequence" :depends-on ("_package"))
  ))